import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import torch.nn as nn

from models.siamese import SiameseNetwork
from utils.dataset import TripletDataset
from config import *


def train():
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

   model = SiameseNetwork().to(DEVICE)

   criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

   optimizer = torch.optim.Adam(
      model.parameters(),
      lr=LEARNING_RATE
   )

   best_loss = float("inf")

   for epoch in range(NUM_EPOCHS):
      model.train()
      running_loss = 0.0

      progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
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

            progress_bar.set_postfix(
               loss=f"{loss.item():.4f}"
            )

      epoch_loss = running_loss / len(train_loader)

      print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {epoch_loss:.4f}"
      )

      if epoch_loss < best_loss:
            best_loss = epoch_loss

            torch.save(
               model.state_dict(),
               f"{CHECKPOINT_DIR}/siamese_best.pth"
            )

            print("→ Best model saved successfully!")

   print("Training completed.")