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
# Threshold Calibration
# ===============================
def calibrate_thresholds(model, loader, device):
   model.eval()

   same_dist, diff_dist = [], []

   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating"):
            img1, img2 = img1.to(device), img2.to(device)

            e1, e2 = model(img1, img2)
            dist = torch.norm(e1 - e2, dim=1)

            label = label.cpu().numpy()

            for d, l in zip(dist.cpu().numpy(), label):
               if l == 1:
                  same_dist.append(d)
               else:
                  diff_dist.append(d)

   same_dist = np.array(same_dist)
   diff_dist = np.array(diff_dist)

   th_same = np.mean(same_dist) + np.std(same_dist)
   th_twin = np.mean(diff_dist) - np.std(diff_dist)

   if th_twin <= th_same:
      mid = (th_same + th_twin) / 2
      th_same = mid - 0.1
      th_twin = mid + 0.1

   model.train()
   return float(th_same), float(th_twin)

# ===============================
# Checkpoint Loader (SAFE)
# ===============================
def load_checkpoint(model, optimizer, checkpoint_dir, device):
   path = os.path.join(checkpoint_dir, "siamese_best.pth")

   if not os.path.exists(path):
      print(" No checkpoint found")
      return 0, float('inf'), 0.35, 0.60

   print(f" Loading: {path}")
   checkpoint = torch.load(path, map_location=device)

   model.load_state_dict(checkpoint["model_state_dict"])

   if optimizer and "optimizer_state_dict" in checkpoint:
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

   start_epoch = checkpoint.get("epoch", -1) + 1
   best_loss = checkpoint.get("best_loss", 999)

   th_same = checkpoint.get("threshold_same", 0.35)
   th_twin = checkpoint.get("threshold_twin", 0.60)

   print(f"✔ Resume epoch {start_epoch}")
   return start_epoch, best_loss, th_same, th_twin


# ===============================
# TRAIN
# ===============================
def train():
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
   ])

   print(" Loading dataset...")
   train_dataset = PairDataset(f"{DATA_DIR}/train", transform)
   val_dataset = PairDataset(f"{DATA_DIR}/val", transform)

   train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
   val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

   model = SiameseNetwork().to(DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   criterion = ContrastiveLoss(margin=MARGIN)

   start_epoch, best_loss, th_same, th_twin = load_checkpoint(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )

   TOTAL_EPOCHS = start_epoch + 20

   print(f"\n🚀 Training {start_epoch} → {TOTAL_EPOCHS}")

   for epoch in range(start_epoch, TOTAL_EPOCHS):

      model.train()
      total_loss = 0

      print(f"\n📅 Epoch {epoch+1}")

      for img1, img2, label in tqdm(train_loader):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

      epoch_loss = total_loss / len(train_loader)
      print(f"📉 Loss: {epoch_loss:.4f}")

      # SAVE BEST
      if epoch_loss < best_loss:
            best_loss = epoch_loss

            print("✨ New Best Model")

            th_same, th_twin = calibrate_thresholds(model, val_loader, DEVICE)

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

            print(f"🎯 same={th_same:.3f} twin={th_twin:.3f}")

      if (epoch + 1) % 5 == 0:
            try:
               dummy1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
               dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

               onnx_path = os.path.join(CHECKPOINT_DIR, f"model_{epoch+1}.onnx")

               torch.onnx.export(
                  model,
                  (dummy1, dummy2),
                  onnx_path,
                  input_names=['img1','img2'],
                  output_names=['out1','out2'],
                  opset_version=11
               )

               print("📦 ONNX exported")
            except Exception as e:
               print(f"⚠️ ONNX skipped: {e}")

   print("\n✅ Training finished")
   print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
   train()