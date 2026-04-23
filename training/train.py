# training/train.py - Simple auto-resume version
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import glob

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *

def find_best_checkpoint(checkpoint_dir):
   """Find the best checkpoint (lowest loss)"""
   best_file = None
   best_loss = float('inf')
   
   # Check for best model first
   best_path = os.path.join(checkpoint_dir, "siamese_best.pth")
   if os.path.exists(best_path):
      return best_path
   
   # Otherwise find checkpoint with lowest loss
   checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
   
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

def load_best_model(model, optimizer, checkpoint_dir, device):
   """Load the best model automatically"""
   checkpoint_path = find_best_checkpoint(checkpoint_dir)
   
   if not checkpoint_path:
      print("No checkpoint found. Starting from scratch.")
      return 0, float('inf'), 0.4, 0.7
   
   print(f"Loading best model: {os.path.basename(checkpoint_path)}")
   checkpoint = torch.load(checkpoint_path, map_location=device)
   
   # Load model
   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)
   
   # Load optimizer if available
   if optimizer and "optimizer_state_dict" in checkpoint:
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
   
   # Get info
   epoch = checkpoint.get("epoch", 0)
   best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float('inf')))
   th_same = checkpoint.get("threshold_same", 0.4)
   th_twin = checkpoint.get("threshold_twin", 0.7)
   
   print(f"Resuming from epoch {epoch+1} (best loss: {best_loss:.4f})")
   print(f"Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   
   return epoch + 1, best_loss, th_same, th_twin

def train():
   """Train with automatic resume from best checkpoint"""
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
   
   # Load best model automatically
   start_epoch, best_loss, th_same, th_twin = load_best_model(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )
   
   print(f"\n{'='*50}")
   print(f"Training on {DEVICE}")
   print(f"Epochs: {start_epoch+1} to {NUM_EPOCHS}")
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
      
      # Save checkpoint
      checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
            "threshold_same": th_same,
            "threshold_twin": th_twin
      }
      
      # Save epoch checkpoint
      torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
      
      # Save if best
      if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
            print(f"✨ New best model! Loss: {epoch_loss:.4f}")
      
      # Keep latest checkpoint
      torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_latest.pth"))
   
   print(f"\n{'='*50}")
   print("Training completed!")
   print(f"Best loss: {best_loss:.4f}")
   print(f"Final thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   print(f"{'='*50}")

if __name__ == "__main__":
   train()