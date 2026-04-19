import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *

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

   dataset = PairDataset(f"{DATA_DIR}/train", transform)
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

   best_loss = float("inf")

   for epoch in range(NUM_EPOCHS):

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
      print(f"Epoch {epoch+1}: {epoch_loss:.4f}")
      
      th_same = 0.4
      th_twin = 0.7

      torch.save({
            "model_state_dict": model.state_dict(),
            "threshold_same": th_same,
            "threshold_twin": th_twin
      }, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))

   print("Training done")