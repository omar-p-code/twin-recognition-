# training/train.py - Fixed version
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


# ===============================
# Threshold Calibration - FIXED
# ===============================
def calibrate_thresholds(model, loader, device):
   model.eval()
   same_dist = []
   diff_dist = []

   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            e1, e2 = model(img1, img2)
            dist = torch.norm(e1 - e2, dim=1)

            # Move to CPU for numpy
            dist_np = dist.cpu().numpy()
            label_np = label.cpu().numpy()

            for d, l in zip(dist_np, label_np):
               if l == 1:  # Same person
                  same_dist.append(float(d))
               else:  # Different person
                  diff_dist.append(float(d))

   if len(same_dist) == 0 or len(diff_dist) == 0:
      print("⚠️  Not enough data for calibration, using defaults")
      model.train()
      return 0.35, 0.60

   same_dist = np.array(same_dist)
   diff_dist = np.array(diff_dist)

   # Use percentiles instead of mean+std (more robust)
   th_same = np.percentile(same_dist, 95)  # 95% of same pairs below this
   th_twin = np.percentile(diff_dist, 5)   # 5% of diff pairs below this

   # Safety caps
   th_same = min(th_same, 0.35)
   th_twin = max(th_twin, 0.50)

   # Ensure gap
   if th_twin - th_same < 0.10:
      mid = (th_same + th_twin) / 2
      th_same = mid - 0.08
      th_twin = mid + 0.08

   print(f"  Same: mean={same_dist.mean():.4f}, std={same_dist.std():.4f}")
   print(f"  Diff: mean={diff_dist.mean():.4f}, std={diff_dist.std():.4f}")
   print(f"  Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")

   model.train()
   return float(th_same), float(th_twin)


# ===============================
# Checkpoint Loader
# ===============================
def load_checkpoint(model, optimizer, checkpoint_dir, device):
   path = os.path.join(checkpoint_dir, "siamese_best.pth")

   if not os.path.exists(path):
      print("📂 No checkpoint found, starting from scratch")
      return 0, float('inf'), 0.35, 0.60

   print(f"📂 Loading: {path}")
   checkpoint = torch.load(path, map_location=device)

   # Load model state
   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)

   # Load optimizer if available
   if optimizer and "optimizer_state_dict" in checkpoint:
      try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      except:
            print("⚠️  Could not load optimizer state")

   start_epoch = checkpoint.get("epoch", -1) + 1
   best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float('inf')))
   th_same = checkpoint.get("threshold_same", 0.35)
   th_twin = checkpoint.get("threshold_twin", 0.60)

   print(f"✓ Resuming from epoch {start_epoch}")
   print(f"  Best loss: {best_loss:.4f}")
   print(f"  Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")

   return start_epoch, best_loss, th_same, th_twin


# ===============================
# TRAIN
# ===============================
def train():
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

   # Data transforms
   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ColorJitter(brightness=0.1, contrast=0.1),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

   # Validation transform (no augmentation)
   val_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

   print("📦 Loading dataset...")
   train_dataset = PairDataset(f"{DATA_DIR}/train", transform)
   val_dataset = PairDataset(f"{DATA_DIR}/val", val_transform)

   train_loader = DataLoader(
      train_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=True, 
      num_workers=0  # Set to 0 to avoid multiprocessing issues
   )
   val_loader = DataLoader(
      val_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=False, 
      num_workers=0
   )

   print(f"✓ Train: {len(train_dataset)} pairs, Val: {len(val_dataset)} pairs")

   # Model
   model = SiameseNetwork().to(DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   criterion = ContrastiveLoss(margin=MARGIN)

   # Load checkpoint
   start_epoch, best_loss, th_same, th_twin = load_checkpoint(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )

   TOTAL_EPOCHS = start_epoch + 20

   print(f"\n{'='*50}")
   print(f"🚀 Training epochs {start_epoch+1} → {TOTAL_EPOCHS}")
   print(f"  Device: {DEVICE}")
   print(f"  Learning rate: {LEARNING_RATE}")
   print(f"  Batch size: {BATCH_SIZE}")
   print(f"{'='*50}\n")

   for epoch in range(start_epoch, TOTAL_EPOCHS):
      model.train()
      total_loss = 0

      print(f"\n📅 Epoch {epoch+1}/{TOTAL_EPOCHS}")

      progress_bar = tqdm(train_loader, desc="Training")
      for img1, img2, label in progress_bar:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

      epoch_loss = total_loss / len(train_loader)
      print(f"  📊 Average Loss: {epoch_loss:.4f}")

      # Save if best
      if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("  ✨ New best model!")

            # Calibrate thresholds
            print("  🎯 Calibrating thresholds...")
            th_same, th_twin = calibrate_thresholds(model, val_loader, DEVICE)

            # Save PyTorch checkpoint
            checkpoint = {
               "epoch": epoch,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "loss": epoch_loss,
               "best_loss": best_loss,
               "threshold_same": th_same,
               "threshold_twin": th_twin
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
            
            try:
               model.eval()
               dummy1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
               dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
               
               onnx_path = os.path.join(CHECKPOINT_DIR, "siamese_best.onnx")
               torch.onnx.export(
                  model,
                  (dummy1, dummy2),
                  onnx_path,
                  input_names=['img1', 'img2'],
                  output_names=['out1', 'out2'],
                  opset_version=11,
                  dynamic_axes={
                        'img1': {0: 'batch_size'},
                        'img2': {0: 'batch_size'},
                        'out1': {0: 'batch_size'},
                        'out2': {0: 'batch_size'}
                  }
               )
               model.train()
               print(f"  📦 ONNX exported: {onnx_path}")
            except Exception as e:
               print(f"  ⚠️ ONNX export skipped: {e}")
               model.train()
            
            print(f"  💾 Checkpoint saved")

      if (epoch + 1) % 5 == 0:
            checkpoint = {
               "epoch": epoch,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "loss": epoch_loss,
               "best_loss": best_loss,
               "threshold_same": th_same,
               "threshold_twin": th_twin
            }
            torch.save(
               checkpoint, 
               os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth")
            )

   print(f"\n{'='*50}")
   print(f"✅ Training completed!")
   print(f"  Epochs: {start_epoch+1} to {TOTAL_EPOCHS}")
   print(f"  Best loss: {best_loss:.4f}")
   print(f"  Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   print(f"{'='*50}")


if __name__ == "__main__":
   train()