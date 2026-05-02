# training/train.py - Minimal training with auto-resume
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *


def calibrate_thresholds(model, loader, device):
   model.eval()

   same_dist = []
   diff_dist = []

   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating"):

            img1 = img1.to(device)
            img2 = img2.to(device)

            e1, e2 = model(img1, img2)

            distances = torch.nn.functional.pairwise_distance(e1, e2)

            for d, l in zip(distances.cpu().numpy(), label.numpy()):

               if l == 1:
                  same_dist.append(float(d))
               else:
                  diff_dist.append(float(d))

   same_dist = np.array(same_dist)
   diff_dist = np.array(diff_dist)

   # Statistics
   same_mean = np.mean(same_dist)
   same_std = np.std(same_dist)

   diff_mean = np.mean(diff_dist)
   diff_std = np.std(diff_dist)

   # Main threshold between classes
   threshold = (same_mean + diff_mean) / 2

   # Optional tighter same threshold
   th_same = same_mean + same_std * 0.5

   # Twin threshold = center split
   th_twin = threshold

   print(
      f"\n📊 Calibration Stats:"
      f"\n  SAME  -> mean={same_mean:.4f} std={same_std:.4f}"
      f"\n  DIFF  -> mean={diff_mean:.4f} std={diff_std:.4f}"
      f"\n  FINAL -> same={th_same:.4f} twin={th_twin:.4f}"
   )

   model.train()

   return float(th_same), float(th_twin)

def load_checkpoint(model, optimizer, checkpoint_dir, device):
   """Load checkpoint for resume training"""
   path = os.path.join(checkpoint_dir, "checkpoint.pth")

   if not os.path.exists(path):
      print("No checkpoint found, starting from scratch")
      return 0, float('inf'), 0.35, 0.60

   ckpt = torch.load(path, map_location=device)
   model.load_state_dict(ckpt["model_state_dict"])

   if optimizer and "optimizer_state_dict" in ckpt:
      try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
      except:
            print("Warning: Could not load optimizer state")

   start_epoch = ckpt.get("epoch", -1) + 1
   best_loss = ckpt.get("best_loss", float('inf'))
   th_same = ckpt.get("threshold_same", 0.35)
   th_twin = ckpt.get("threshold_twin", 0.60)

   print(f"Resumed from epoch {start_epoch} | best_loss={best_loss:.4f} "
         f"| same={th_same:.4f} twin={th_twin:.4f}")
   return start_epoch, best_loss, th_same, th_twin


def validate(model, loader, criterion, device):
   """Run validation and return average loss"""
   model.eval()
   total_loss = 0

   with torch.no_grad():
      for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            total_loss += loss.item()

   model.train()
   return total_loss / len(loader)


def save_checkpoint(model, epoch, loss, best_loss, th_same, th_twin, checkpoint_dir):
   """Save single checkpoint file (overwrites)"""
   ckpt = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "loss": loss,
      "best_loss": best_loss,
      "threshold_same": th_same,
      "threshold_twin": th_twin
   }
   torch.save(ckpt, os.path.join(checkpoint_dir, "checkpoint.pth"))


def export_onnx(model, checkpoint_dir, device):
   """Export model to ONNX format"""
   try:
      model.eval()
      dummy1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
      dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

      torch.onnx.export(
            model,
            (dummy1, dummy2),
            os.path.join(checkpoint_dir, "model.onnx"),
            input_names=['img1', 'img2'],
            output_names=['out1', 'out2'],
            opset_version=17,
            dynamo=False
      )
      model.train()
      return True
   except Exception as e:
      print(f"ONNX export failed: {e}")
      model.train()
      return False


def train():
   """Main training loop"""
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

   # Data
   train_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ColorJitter(
         brightness=0.2,
         contrast=0.2,
         saturation=0.2
      ),
      transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
      transforms.ToTensor(),
      transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
   ])

   val_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
   ])

   train_dataset = PairDataset(f"{DATA_DIR}/train", train_transform)
   val_dataset = PairDataset(f"{DATA_DIR}/val", val_transform)

   train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
   val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

   print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

   # Model
   model = SiameseNetwork().to(DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   criterion = ContrastiveLoss(margin=MARGIN)

   # Resume
   start_epoch, best_loss, th_same, th_twin = load_checkpoint(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )

   TOTAL_EPOCHS = start_epoch + 20
   print(f"Training {start_epoch+1}-{TOTAL_EPOCHS} | Device: {DEVICE} | LR: {LEARNING_RATE}")

   for epoch in range(start_epoch, TOTAL_EPOCHS):
      # Train
      model.train()
      train_loss = 0

      pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
      for img1, img2, label in pbar:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

      train_loss /= len(train_loader)

      # Validate
      val_loss = validate(model, val_loader, criterion, DEVICE)
      print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

      # Save best (using validation loss)
      if val_loss < best_loss:
            best_loss = val_loss
            print(f"  New best model! (val_loss={val_loss:.4f})")

            th_same, th_twin = calibrate_thresholds(model, val_loader, DEVICE)
            save_checkpoint(model, epoch, train_loss, best_loss, th_same, th_twin, CHECKPOINT_DIR)
            export_onnx(model, CHECKPOINT_DIR, DEVICE)

   print(f"\nDone | Best val_loss: {best_loss:.4f} | same={th_same:.4f} twin={th_twin:.4f}")
   print(f"Files: checkpoint.pth, model.onnx")


if __name__ == "__main__":
   train()