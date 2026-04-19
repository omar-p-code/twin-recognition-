import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *
from utils import onnx_export

def calibrate_thresholds(model, loader, device):

   model.eval()

   same_dist = []
   diff_dist = []

   with torch.no_grad():
      for img1, img2, label in loader:

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

   same_dist = np.array(same_dist)
   diff_dist = np.array(diff_dist)

   # simple but effective split
   th_same = np.percentile(same_dist, 90)
   th_twin = np.percentile(diff_dist, 10)

   return float(th_same), float(th_twin)

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
      
      th_same, th_twin = calibrate_thresholds(model, loader, DEVICE)

      torch.save({
            "model_state_dict": model.state_dict(),
            "threshold_same": th_same,
            "threshold_twin": th_twin
      }, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
      
      onnx_export(model, os.path.join(CHECKPOINT_DIR, "siamese.onnx"))

   print("Training done")