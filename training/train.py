import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import torch.nn as nn

from models.siamese import SiameseNetwork
from utils.dataset import TripletDataset
from config import *


def load_checkpoint(path, model, optimizer=None):
   """
   Safe checkpoint loader that supports old and new formats.
   """

   checkpoint = torch.load(path, map_location=DEVICE)

   # Case 1: new checkpoint format (dict with keys)
   if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"], strict=False)

      if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

      epoch = checkpoint.get("epoch", -1)
      best_loss = checkpoint.get("best_loss", float("inf"))

      return epoch, best_loss

   # Case 2: raw state_dict (old format)
   elif isinstance(checkpoint, dict):
      model.load_state_dict(checkpoint, strict=False)
      return -1, float("inf")

   else:
      raise ValueError("Unsupported checkpoint format")


def train():

   # ================== Create checkpoint folder ==================
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

   # ================== Data augmentation ==================
   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(brightness=0.2, contrast=0.2),
      transforms.ToTensor(),
      transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
      )
   ])

   # ================== Dataset ==================
   train_dataset = TripletDataset(
      root_dir=f"{DATA_DIR}/train",
      transform=transform
   )

   train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=0,
      pin_memory=False
   )

   # ================== Model ==================
   model = SiameseNetwork().to(DEVICE)

   # ================== Loss function ==================
   criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

   # ================== Optimizer ==================
   optimizer = torch.optim.Adam(
      model.parameters(),
      lr=LEARNING_RATE
   )

   # ================== Checkpoint setup ==================
   checkpoint_path = os.path.join(CHECKPOINT_DIR, "siamese_last.pth")

   start_epoch = 0
   best_loss = float("inf")

   # ================== Load checkpoint if exists ==================
   if os.path.exists(checkpoint_path):
      print("Loading checkpoint...")

      start_epoch, best_loss = load_checkpoint(
            checkpoint_path,
            model,
            optimizer
      )

      start_epoch += 1
      print(f"Resuming from epoch {start_epoch}")

   # ================== Dynamic training length ==================
   TOTAL_EPOCHS = start_epoch + 20

   # ================== Training loop ==================
   for epoch in range(start_epoch, TOTAL_EPOCHS):

      model.train()
      running_loss = 0.0

      progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}"
      )

      for anchor, positive, negative in progress_bar:

            anchor = anchor.to(DEVICE)
            positive = positive.to(DEVICE)
            negative = negative.to(DEVICE)

            optimizer.zero_grad()

            anchor_emb, positive_emb, negative_emb = model(
               anchor,
               positive,
               negative
            )

            loss = criterion(
               anchor_emb,
               positive_emb,
               negative_emb
            )

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

      # ================== Epoch loss ==================
      epoch_loss = running_loss / len(train_loader)

      print(f"\nEpoch [{epoch+1}/{TOTAL_EPOCHS}] Loss: {epoch_loss:.4f}")

      # ================== Save last checkpoint ==================
      torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss
      }, checkpoint_path)

      # ================== Save best model ==================
      if epoch_loss < best_loss:
            best_loss = epoch_loss

            torch.save({
               "epoch": epoch,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "best_loss": best_loss
            }, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))

            print("Best model saved!")

   print("Training completed.")