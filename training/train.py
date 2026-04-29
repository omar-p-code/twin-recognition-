import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import glob
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *

def calibrate_thresholds(model, loader, device):
   """Calibrate thresholds based on current model"""
   model.eval()
   same_dist = []
   diff_dist = []

   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating thresholds"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            dist = torch.nn.functional.pairwise_distance(out1, out2)

            for d, l in zip(dist.cpu().numpy(), label.cpu().numpy()):
               if l == 1:
                  same_dist.append(float(d))
               else:
                  diff_dist.append(float(d))

   same_dist = np.array(same_dist)
   diff_dist = np.array(diff_dist)
   
   if len(same_dist) > 0:
      th_same = np.percentile(same_dist, 95)
   else:
      th_same = 0.35

   if len(diff_dist) > 0:
      th_twin = np.percentile(diff_dist, 5)
   else:
      th_twin = 0.55

   th_same = min(th_same, 0.35)
   th_twin = max(th_twin, 0.50)
   
   if th_twin - th_same < 0.10:
      th_same = min(th_same, 0.30)
      th_twin = max(th_twin, 0.45)

   model.train()
   return float(th_same), float(th_twin)


def find_best_checkpoint(checkpoint_dir):
   """Find the best checkpoint"""
   best_path = os.path.join(checkpoint_dir, "siamese_best.pth")
   if os.path.exists(best_path):
      return best_path
   
   # Find latest checkpoint
   checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
   if not checkpoint_files:
      return None
   
   # Find the one with lowest loss
   best_file = None
   best_loss = float('inf')
   for f in checkpoint_files:
      try:
            checkpoint = torch.load(f, map_location='cpu')
            loss = checkpoint.get('loss', float('inf'))
            if loss < best_loss:
               best_loss = loss
               best_file = f
      except:
            continue
   
   return best_file


def load_checkpoint(model, optimizer, checkpoint_dir, device):
   """Load checkpoint and return start epoch, best loss, thresholds"""
   checkpoint_path = find_best_checkpoint(checkpoint_dir)
   
   if not checkpoint_path:
      print("No checkpoint found. Starting from scratch.")
      return 0, float('inf'), 0.35, 0.55
   
   print(f"📂 Loading checkpoint: {os.path.basename(checkpoint_path)}")
   checkpoint = torch.load(checkpoint_path, map_location=device)
   
   # Load model
   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)
   
   # Load optimizer
   if optimizer and "optimizer_state_dict" in checkpoint:
      try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      except:
            print("⚠️  Could not load optimizer state, using fresh optimizer")
   
   # Get training info
   start_epoch = checkpoint.get("epoch", -1) + 1
   best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float('inf')))
   th_same = checkpoint.get("threshold_same", 0.35)
   th_twin = checkpoint.get("threshold_twin", 0.55)
   
   print(f"✓ Resuming from epoch {start_epoch + 1}")
   print(f"  Best loss so far: {best_loss:.4f}")
   print(f"  Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   
   return start_epoch, best_loss, th_same, th_twin


def train():
   """Train with auto-resume and dynamic total epochs"""
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)
   
   # Data transforms
   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
      )
   ])
   
   # Load dataset
   print("📦 Loading dataset...")
   dataset = PairDataset(f"{DATA_DIR}/train", transform)
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   print(f"✓ Dataset loaded: {len(dataset)} pairs")
   
   # Initialize model
   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss(margin=MARGIN)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   
   # Load checkpoint - THIS IS THE AUTO-RESUME
   start_epoch, best_loss, th_same, th_twin = load_checkpoint(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )
   
   # DYNAMIC TOTAL EPOCHS: start + 20
   TOTAL_EPOCHS = start_epoch + 20
   
   print(f"\n{'='*50}")
   print(f"🚀 Training Configuration")
   print(f"  Device: {DEVICE}")
   print(f"  Start epoch: {start_epoch + 1}")
   print(f"  Total epochs: {TOTAL_EPOCHS} (start + 20)")
   print(f"  Epochs to train: {TOTAL_EPOCHS - start_epoch}")
   print(f"  Best loss so far: {best_loss:.4f}")
   print(f"{'='*50}\n")
   
   # Training loop
   for epoch in range(start_epoch, TOTAL_EPOCHS):
      model.train()
      total_loss = 0
      
      print(f"\n📅 Epoch {epoch+1}/{TOTAL_EPOCHS}")
      
      progress_bar = tqdm(loader, desc="Training")
      for img1, img2, label in progress_bar:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
      
      epoch_loss = total_loss / len(loader)
      print(f"  📊 Average Loss: {epoch_loss:.4f}")
      
      # Save checkpoint every epoch
      checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
            "threshold_same": th_same,
            "threshold_twin": th_twin
      }
      
      # Save epoch checkpoint
      # torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
      
      # Save if best
      if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
            print(f"  ✨ New best model! Loss: {epoch_loss:.4f}")
            
            # Calibrate thresholds on best model
            print("  🎯 Calibrating thresholds...")
            th_same, th_twin = calibrate_thresholds(model, loader, DEVICE)
            
            # Update checkpoint with new thresholds
            checkpoint["threshold_same"] = th_same
            checkpoint["threshold_twin"] = th_twin
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
      
      # Always save latest
      torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_latest.pth"))
      
      # onnx export every 5 epochs
      if (epoch + 1) % 5 == 0:
            dummy_input1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            dummy_input2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            onnx_path = os.path.join(CHECKPOINT_DIR, f"siamese_epoch_{epoch+1}.onnx")
            torch.onnx.export(
                  model,
                  (dummy_input1, dummy_input2),
                  onnx_path,
                  input_names=['img1', 'img2'],
                  output_names=['out1', 'out2'],
                  opset_version=11
            )
            print(f"  📦 Exported ONNX model: {onnx_path}")
   
   print(f"\n{'='*50}")
   print(f"✅ Training completed!")
   print(f"  Epochs trained: {start_epoch+1} to {TOTAL_EPOCHS}")
   print(f"  Best loss: {best_loss:.4f}")
   print(f"  Final thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   print(f"  Model saved: {CHECKPOINT_DIR}/siamese_best.pth")
   print(f"{'='*50}")


if __name__ == "__main__":
   train()