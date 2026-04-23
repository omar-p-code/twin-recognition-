# training/train.py - Simplified version without scheduler issues
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import glob

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *
from utils.export_onnx import export_onnx

def find_latest_checkpoint(checkpoint_dir):
   """Find the latest checkpoint in the directory"""
   checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
   
   if not checkpoint_files:
      return None
   
   # Look for epoch checkpoints
   epoch_checkpoints = [f for f in checkpoint_files if "epoch_" in f]
   if epoch_checkpoints:
      def get_epoch_num(path):
            try:
               basename = os.path.basename(path)
               epoch_str = basename.split("epoch_")[1].split(".")[0]
               return int(epoch_str)
            except:
               return -1
      
      epoch_checkpoints.sort(key=get_epoch_num, reverse=True)
      return epoch_checkpoints[0]
   
   # Fall back to best model
   best_model = os.path.join(checkpoint_dir, "siamese_best.pth")
   if os.path.exists(best_model):
      return best_model
   
   return None

def load_checkpoint(model, optimizer, checkpoint_path, device):
   """Load checkpoint and return starting epoch, best_loss, and thresholds"""
   if not os.path.exists(checkpoint_path):
      print("No checkpoint found. Starting from scratch.")
      return 0, float("inf"), 0.4, 0.7
   
   print(f"📂 Loading checkpoint: {checkpoint_path}")
   checkpoint = torch.load(checkpoint_path, map_location=device)
   
   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)
   
   if optimizer and "optimizer_state_dict" in checkpoint:
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      print("  ✓ Optimizer state restored")
   
   start_epoch = checkpoint.get("epoch", 0)
   if start_epoch > 0:
      start_epoch += 1
   
   best_loss = checkpoint.get("best_loss", float("inf"))
   th_same = checkpoint.get("threshold_same", 0.4)
   th_twin = checkpoint.get("threshold_twin", 0.7)
   
   print(f"  ✓ Resuming from epoch {start_epoch}")
   print(f"  ✓ Best loss: {best_loss:.4f}")
   print(f"  ✓ Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   
   return start_epoch, best_loss, th_same, th_twin

def calibrate_thresholds(model, loader, device):
   model.eval()
   same_dist = []
   diff_dist = []

   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating"):
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            out1, out2 = model(img1, img2)
            dist = torch.nn.functional.pairwise_distance(out1, out2)

            for d, l in zip(dist.cpu().numpy(), label.cpu().numpy()):
               if l == 1:
                  same_dist.append(d)
               else:
                  diff_dist.append(d)

   th_same = np.percentile(same_dist, 90) if len(same_dist) > 0 else 0.4
   th_twin = np.percentile(diff_dist, 10) if len(diff_dist) > 0 else 0.7

   return float(th_same), float(th_twin)

def train(resume=True):
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
      )
   ])

   print("📁 Loading dataset...")
   dataset = PairDataset(f"{DATA_DIR}/train", transform)
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   print(f"✓ Dataset loaded: {len(dataset)} pairs")

   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss(margin=MARGIN)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   # Try to resume
   start_epoch = 0
   best_loss = float("inf")
   th_same = 0.4
   th_twin = 0.7
   
   if resume:
      latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
      if latest_checkpoint:
            start_epoch, best_loss, th_same, th_twin = load_checkpoint(
               model, optimizer, latest_checkpoint, DEVICE
            )
      else:
            print("No checkpoint found. Starting fresh.")

   print(f"\n{'='*50}")
   print(f"Starting Training")
   print(f"Device: {DEVICE}")
   print(f"Epochs: {NUM_EPOCHS}")
   print(f"Start: {start_epoch + 1}")
   print(f"{'='*50}\n")

   for epoch in range(start_epoch, NUM_EPOCHS):
      model.train()
      total_loss = 0
      
      print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
      
      for img1, img2, label in tqdm(loader):
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            label = label.to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

      epoch_loss = total_loss / len(loader)
      print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
      
      # Calibrate every 5 epochs
      if (epoch + 1) % 5 == 0:
            th_same, th_twin = calibrate_thresholds(model, loader, DEVICE)
            print(f"New thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
      
      # Save checkpoint
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
      torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
      
      # Save best model
      if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
            print(f"New best model! (loss: {epoch_loss:.4f})")
      
      # Export ONNX
      if (epoch + 1) % 5 == 0:
            try:
               export_onnx(model, os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.onnx"))
            except:
               pass

   print("Training completed!")

if __name__ == "__main__":
   train(resume=True)