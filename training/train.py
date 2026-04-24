# training/train.py

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np

import torch.onnx

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *

# ================== PATHS ==================
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
ONNX_FILE = os.path.join(CHECKPOINT_DIR, "model.onnx")
ONNX_QUANT_FILE = os.path.join(CHECKPOINT_DIR, "model_quant.onnx")


# ================== LOAD CHECKPOINT ==================
def load_checkpoint(model, optimizer):
   if os.path.exists(CHECKPOINT_FILE):
      print(f"Loading checkpoint: {CHECKPOINT_FILE}")
      checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)

      model.load_state_dict(checkpoint["model_state_dict"])
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

      start_epoch = checkpoint["epoch"] + 1
      best_loss = checkpoint["best_loss"]
      th_same = checkpoint.get("threshold_same", 0.4)
      th_twin = checkpoint.get("threshold_twin", 0.7)

      # 🔥 هنا التعديل: نزود 20 epoch
      new_total_epochs = start_epoch + 20

      print(f"Resumed from epoch {start_epoch}")
      print(f"Training will continue to epoch {new_total_epochs}")
      print(f"Best loss: {best_loss:.4f}")
      print(f"Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")

      return start_epoch, new_total_epochs, best_loss, th_same, th_twin

   else:
      print("No checkpoint found. Starting from scratch.")
      return 0, NUM_EPOCHS, float("inf"), 0.4, 0.7


# ================== SAVE CHECKPOINT ==================
def save_checkpoint(model, optimizer, epoch, loss, best_loss, th_same, th_twin):
   checkpoint = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": loss,
      "best_loss": best_loss,
      "threshold_same": th_same,
      "threshold_twin": th_twin
   }

   torch.save(checkpoint, CHECKPOINT_FILE)
   print(f"💾 Checkpoint saved (epoch {epoch+1})")


# ================== CALIBRATE ==================
def calibrate_thresholds(model, loader, device):
   model.eval()
   same_dist = []
   diff_dist = []

   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            out1, out2 = model(img1, img2)
            dist = torch.nn.functional.pairwise_distance(out1, out2)

            for d, l in zip(dist.cpu().numpy(), label.cpu().numpy()):
               if l == 1:
                  same_dist.append(d)
               else:
                  diff_dist.append(d)

   th_same = np.percentile(same_dist, 90) if same_dist else 0.4
   th_twin = np.percentile(diff_dist, 10) if diff_dist else 0.7

   model.train()
   return float(th_same), float(th_twin)


# ================== ONNX EXPORT ==================
def export_onnx(model):
   model.eval()

   dummy1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
   dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

   torch.onnx.export(
      model,
      (dummy1, dummy2),
      ONNX_FILE,
      export_params=True,
      opset_version=11,
      do_constant_folding=True,
      input_names=["img1", "img2"],
      output_names=["emb1", "emb2"]
   )

   print(f"✅ ONNX saved: {ONNX_FILE}")


# ================== QUANTIZATION ==================
def quantize_onnx():
   try:
      from onnxruntime.quantization import quantize_dynamic, QuantType

      quantize_dynamic(
            ONNX_FILE,
            ONNX_QUANT_FILE,
            weight_type=QuantType.QInt8
      )

      print(f"🔥 Quantized model saved: {ONNX_QUANT_FILE}")

   except Exception as e:
      print("⚠️ Quantization failed:", e)


# ================== TRAIN ==================
def train():
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

   print("Loading dataset...")
   dataset = PairDataset(f"{DATA_DIR}/train", transform)
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   print(f"Dataset size: {len(dataset)}")

   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss(margin=MARGIN)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   start_epoch, total_epochs, best_loss, th_same, th_twin = load_checkpoint(model, optimizer)

   print(f"\nTraining from {start_epoch} to {total_epochs}\n")

   for epoch in range(start_epoch, total_epochs):
      model.train()
      total_loss = 0

      print(f"\nEpoch {epoch+1}/{total_epochs}")

      for img1, img2, label in tqdm(loader):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

      epoch_loss = total_loss / len(loader)
      print(f"Loss: {epoch_loss:.4f}")

      if epoch_loss < best_loss:
            best_loss = epoch_loss
            print("✨ New best model")

      if (epoch + 1) % 10 == 0:
            th_same, th_twin = calibrate_thresholds(model, loader, DEVICE)
            print(f"New thresholds: {th_same:.4f}, {th_twin:.4f}")

      save_checkpoint(model, optimizer, epoch, epoch_loss, best_loss, th_same, th_twin)

   print("\nTraining Done ✅")

   # ================== EXPORT ==================
   export_onnx(model)
   quantize_onnx()


if __name__ == "__main__":
   train()