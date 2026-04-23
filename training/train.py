# training/train.py - COMPLETE WORKING VERSION with single checkpoint
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *

# Single checkpoint file - this is the ONLY file
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")

def load_checkpoint(model, optimizer):
   """Load from single checkpoint file - returns starting epoch"""
   if os.path.exists(CHECKPOINT_FILE):
      print(f"Loading checkpoint: {CHECKPOINT_FILE}")
      checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
      
      model.load_state_dict(checkpoint["model_state_dict"])
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      
      start_epoch = checkpoint["epoch"] + 1
      best_loss = checkpoint["best_loss"]
      th_same = checkpoint.get("threshold_same", 0.4)
      th_twin = checkpoint.get("threshold_twin", 0.7)
      
      print(f"Resumed from epoch {checkpoint['epoch']+1} (loss: {checkpoint['loss']:.4f})")
      print(f"Best loss so far: {best_loss:.4f}")
      print(f"Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
      
      return start_epoch, best_loss, th_same, th_twin
   else:
      print("No checkpoint found. Starting from scratch.")
      return 0, float("inf"), 0.4, 0.7

def save_checkpoint(model, optimizer, epoch, loss, best_loss, th_same, th_twin):
   """Save to single checkpoint file (OVERWRITES the same file)"""
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
   print(f"  💾 Checkpoint saved (epoch {epoch+1})")

def calibrate_thresholds(model, loader, device):
   """Calculate optimal thresholds based on current model performance"""
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
   
   model.train()
   return float(th_same), float(th_twin)

def train():
   """Main training function with single checkpoint file"""
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)
   
   # Data transforms
   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
      )
   ])
   
   # Load dataset
   print("Loading dataset...")
   dataset = PairDataset(f"{DATA_DIR}/train", transform)
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   print(f"Dataset loaded: {len(dataset)} pairs")
   
   # Initialize model
   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss(margin=MARGIN)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   
   # Load from single checkpoint
   start_epoch, best_loss, th_same, th_twin = load_checkpoint(model, optimizer)
   
   print(f"\n{'='*50}")
   print(f"Training on {DEVICE}")
   print(f"Target epochs: {NUM_EPOCHS}")
   print(f"Starting from epoch: {start_epoch + 1}")
   print(f"Best loss so far: {best_loss:.4f}")
   print(f"{'='*50}\n")
   
   # Training loop
   for epoch in range(start_epoch, NUM_EPOCHS):
      model.train()
      total_loss = 0
      
      print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
      
      progress_bar = tqdm(loader, desc="Training")
      for img1, img2, label in progress_bar:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
      
      epoch_loss = total_loss / len(loader)
      print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}")
      
      # Update best loss
      if epoch_loss < best_loss:
            best_loss = epoch_loss
            print(f"  ✨ New best loss: {epoch_loss:.4f}")
      
      # Re-calibrate thresholds every 10 epochs
      if (epoch + 1) % 10 == 0:
            print("  🔧 Re-calibrating thresholds...")
            th_same, th_twin = calibrate_thresholds(model, loader, DEVICE)
            print(f"  New thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
      
      # Save checkpoint (OVERWRITES the same file)
      save_checkpoint(model, optimizer, epoch, epoch_loss, best_loss, th_same, th_twin)
   
   print(f"\n{'='*50}")
   print("Training completed!")
   print(f"Best loss: {best_loss:.4f}")
   print(f"Final thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   print(f"Checkpoint saved: {CHECKPOINT_FILE}")
   print(f"{'='*50}")

if __name__ == "__main__":
   train()