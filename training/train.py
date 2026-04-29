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
#  Threshold Calibration
# ===============================
def calibrate_thresholds(model, loader, device):
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

   th_same = np.percentile(same_dist, 95) if len(same_dist) > 0 else 0.35
   th_twin = np.percentile(diff_dist, 5) if len(diff_dist) > 0 else 0.55

   if th_twin - th_same < 0.1:
      th_same *= 0.9
      th_twin *= 1.1

   model.train()
   return float(th_same), float(th_twin)


# ===============================
# Find Best Checkpoint
# ===============================
def find_best_checkpoint(checkpoint_dir):
   best_path = os.path.join(checkpoint_dir, "siamese_best.pth")
   if os.path.exists(best_path):
      return best_path

   checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
   if not checkpoint_files:
      return None

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


# ===============================
#  Load Checkpoint
# ===============================
def load_checkpoint(model, optimizer, checkpoint_dir, device):
   checkpoint_path = find_best_checkpoint(checkpoint_dir)

   if not checkpoint_path:
      print("No checkpoint found. Starting from scratch.")
      return 0, float('inf'), 0.35, 0.55

   print(f"Loading checkpoint: {os.path.basename(checkpoint_path)}")
   checkpoint = torch.load(checkpoint_path, map_location=device)

   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)

   if optimizer and "optimizer_state_dict" in checkpoint:
      try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      except:
            print("⚠️ Could not load optimizer state")

   start_epoch = checkpoint.get("epoch", -1) + 1
   best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float('inf')))
   th_same = checkpoint.get("threshold_same", 0.35)
   th_twin = checkpoint.get("threshold_twin", 0.55)

   print(f"✓ Resuming from epoch {start_epoch + 1}")
   print(f"Best loss: {best_loss:.4f}")

   return start_epoch, best_loss, th_same, th_twin


# ===============================
#  Training
# ===============================
def train():
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

   print(" Loading dataset...")
   train_dataset = PairDataset(f"{DATA_DIR}/train", transform)
   val_dataset = PairDataset(f"{DATA_DIR}/val", transform)

   train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
   val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

   print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss(margin=MARGIN)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   start_epoch, best_loss, th_same, th_twin = load_checkpoint(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )

   TOTAL_EPOCHS = start_epoch + 20

   for epoch in range(start_epoch, TOTAL_EPOCHS):
      model.train()
      total_loss = 0

      print(f"\n Epoch {epoch+1}/{TOTAL_EPOCHS}")

      for img1, img2, label in tqdm(train_loader):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()

            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

      epoch_loss = total_loss / len(train_loader)
      print(f" Loss: {epoch_loss:.4f}")

      # Save best model
      if epoch_loss < best_loss:
            best_loss = epoch_loss

            print("✨ New best model")

            # Calibrate on validation
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

            print(f"Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")

      # ONNX Export
      if (epoch + 1) % 5 == 0:
            dummy1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
            dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

            onnx_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.onnx")

            torch.onnx.export(
               model,
               (dummy1, dummy2),
               onnx_path,
               input_names=['img1', 'img2'],
               output_names=['out1', 'out2'],
               opset_version=11
            )

            print(f" ONNX saved: {onnx_path}")

   print("\n Training Done")
   print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
   train()