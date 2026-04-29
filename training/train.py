# training/train.py - Complete with better logging
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
# import glob
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *


# ===============================
# Helper Functions
# ===============================
def log(message, level="INFO"):
   """Simple logger with timestamp"""
   timestamp = datetime.now().strftime("%H:%M:%S")
   emoji = {"INFO": "📌", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌", "BEST": "🌟"}
   prefix = emoji.get(level, "•")
   print(f"[{timestamp}] {prefix} {message}")


def calibrate_thresholds(model, loader, device):
   """Calibrate same/twin thresholds using validation data"""
   model.eval()
   same_dist = []
   diff_dist = []

   log("Starting threshold calibration...")
   
   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="Calibrating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            e1, e2 = model(img1, img2)
            dist = torch.norm(e1 - e2, dim=1)

            dist_np = dist.cpu().numpy()
            label_np = label.cpu().numpy()

            for d, l in zip(dist_np, label_np):
               if l == 1:
                  same_dist.append(float(d))
               else:
                  diff_dist.append(float(d))

   if len(same_dist) == 0 or len(diff_dist) == 0:
      log("Not enough data for calibration, using defaults", "WARNING")
      model.train()
      return 0.35, 0.60

   same_dist = np.array(same_dist)
   diff_dist = np.array(diff_dist)

   # Statistics
   same_mean, same_std = same_dist.mean(), same_dist.std()
   diff_mean, diff_std = diff_dist.mean(), diff_dist.std()
   
   # Thresholds using percentiles
   th_same = np.percentile(same_dist, 95)
   th_twin = np.percentile(diff_dist, 5)

   # Safety caps
   th_same = min(th_same, 0.35)
   th_twin = max(th_twin, 0.50)

   # Ensure minimum gap
   if th_twin - th_same < 0.10:
      mid = (th_same + th_twin) / 2
      th_same = mid - 0.08
      th_twin = mid + 0.08

   # Report
   log(f"Same pairs  → mean: {same_mean:.4f}, std: {same_std:.4f}, 95th: {np.percentile(same_dist, 95):.4f}")
   log(f"Diff pairs  → mean: {diff_mean:.4f}, std: {diff_std:.4f}, 5th: {np.percentile(diff_dist, 5):.4f}")
   log(f"Thresholds  → SAME: {th_same:.4f} | TWIN: {th_twin:.4f} | DIFF: > {th_twin:.4f}", "SUCCESS")
   
   # Check overlap
   overlap = (same_dist > th_twin).sum() + (diff_dist < th_same).sum()
   total = len(same_dist) + len(diff_dist)
   log(f"Overlap: {overlap}/{total} samples ({100*overlap/total:.1f}%)")
   
   if overlap / total > 0.15:
      log("High overlap detected - model may need more training", "WARNING")

   model.train()
   return float(th_same), float(th_twin)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
   """Load best checkpoint for resume training"""
   path = os.path.join(checkpoint_dir, "siamese_best.pth")

   if not os.path.exists(path):
      log("No checkpoint found - starting from scratch", "WARNING")
      return 0, float('inf'), 0.35, 0.60

   log(f"Loading checkpoint: {os.path.basename(path)}")
   checkpoint = torch.load(path, map_location=device)

   # Load model weights
   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)

   # Load optimizer state
   if optimizer and "optimizer_state_dict" in checkpoint:
      try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            log("Optimizer state restored")
      except:
            log("Could not restore optimizer state", "WARNING")

   start_epoch = checkpoint.get("epoch", -1) + 1
   best_loss = checkpoint.get("best_loss", checkpoint.get("loss", float('inf')))
   th_same = checkpoint.get("threshold_same", 0.35)
   th_twin = checkpoint.get("threshold_twin", 0.60)

   log(f"Resume from epoch {start_epoch} | Best loss: {best_loss:.4f}")
   log(f"Loaded thresholds → SAME: {th_same:.4f} | TWIN: {th_twin:.4f}")

   return start_epoch, best_loss, th_same, th_twin


def save_checkpoint(model, optimizer, epoch, loss, best_loss, th_same, th_twin, checkpoint_dir):
   """Save PyTorch checkpoint"""
   checkpoint = {
      "epoch": epoch,
      "model_state_dict": model.state_dict(),
      "optimizer_state_dict": optimizer.state_dict(),
      "loss": loss,
      "best_loss": best_loss,
      "threshold_same": th_same,
      "threshold_twin": th_twin
   }
   
   # Save best model
   torch.save(checkpoint, os.path.join(checkpoint_dir, "siamese_best.pth"))
   
   # Also save epoch-specific backup
   torch.save(checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
   
   log(f"Checkpoint saved → epoch {epoch+1} | loss: {loss:.4f}", "SUCCESS")


def export_onnx(model, epoch, checkpoint_dir, device):
   """Export model to ONNX format (for inference only, not resume)"""
   try:
      model.eval()
      dummy1 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
      dummy2 = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
      
      onnx_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.onnx")
      
      torch.onnx.export(
            model,
            (dummy1, dummy2),
            onnx_path,
            input_names=['img1', 'img2'],
            output_names=['out1', 'out2'],
            opset_version=17,
            dynamo=False
      )
      
      # Also save as latest
      latest_onnx = os.path.join(checkpoint_dir, "siamese_best.onnx")
      torch.onnx.export(
            model,
            (dummy1, dummy2),
            latest_onnx,
            input_names=['img1', 'img2'],
            output_names=['out1', 'out2'],
            opset_version=17,
            dynamo=False
      )
      
      model.train()
      log(f"ONNX exported → {onnx_path}", "SUCCESS")
      return True
   except Exception as e:
      log(f"ONNX export failed: {e}", "ERROR")
      model.train()
      return False


# ===============================
# Main Training Function
# ===============================
def train():
   """Main training loop with auto-resume"""
   
   log("="*50)
   log("TRAINING STARTED")
   log("="*50)
   
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)

   # ==================== DATA ====================
   log("Loading datasets...")
   
   train_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(10),
      transforms.ColorJitter(brightness=0.1, contrast=0.1),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

   val_transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

   train_dataset = PairDataset(f"{DATA_DIR}/train", train_transform)
   val_dataset = PairDataset(f"{DATA_DIR}/val", val_transform)

   train_loader = DataLoader(
      train_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=True, 
      num_workers=2,
      pin_memory=True if DEVICE == 'cuda' else False
   )
   val_loader = DataLoader(
      val_dataset, 
      batch_size=BATCH_SIZE, 
      shuffle=False, 
      num_workers=2,
      pin_memory=True if DEVICE == 'cuda' else False
   )

   log(f"Train: {len(train_dataset)} pairs | Val: {len(val_dataset)} pairs")
   log(f"Batches: {len(train_loader)} train | {len(val_loader)} val")

   # ==================== MODEL ====================
   log(f"Initializing model on {DEVICE}...")
   
   model = SiameseNetwork().to(DEVICE)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   criterion = ContrastiveLoss(margin=MARGIN)
   
   total_params = sum(p.numel() for p in model.parameters())
   trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
   log(f"Model parameters: {total_params:,} total | {trainable_params:,} trainable")

   # ==================== RESUME ====================
   start_epoch, best_loss, th_same, th_twin = load_checkpoint(
      model, optimizer, CHECKPOINT_DIR, DEVICE
   )

   TOTAL_EPOCHS = start_epoch + 20

   # ==================== TRAINING ====================
   log(f"{'='*50}")
   log(f"Training: epoch {start_epoch+1} → {TOTAL_EPOCHS}")
   log(f"Device: {DEVICE} | LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Margin: {MARGIN}")
   log(f"Initial thresholds → SAME: {th_same:.4f} | TWIN: {th_twin:.4f}")
   log(f"{'='*50}")

   for epoch in range(start_epoch, TOTAL_EPOCHS):
      model.train()
      total_loss = 0
      
      log(f"Epoch {epoch+1}/{TOTAL_EPOCHS} starting...")
      
      progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
      for batch_idx, (img1, img2, label) in enumerate(progress_bar):
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
               progress_bar.set_postfix({
                  'loss': f'{loss.item():.4f}',
                  'avg': f'{total_loss/(batch_idx+1):.4f}'
               })

      epoch_loss = total_loss / len(train_loader)
      
      # Progress report
      improvement = ""
      if epoch_loss < best_loss:
            improvement = " ↓ NEW BEST!"
            log(f"Loss: {epoch_loss:.4f}{improvement}", "BEST")
            
            # Calibrate on new best
            th_same, th_twin = calibrate_thresholds(model, val_loader, DEVICE)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_loss, th_same, th_twin, CHECKPOINT_DIR)
            
            # Export ONNX
            export_onnx(model, epoch, CHECKPOINT_DIR, DEVICE)
            
      else:
            diff = epoch_loss - best_loss
            log(f"Loss: {epoch_loss:.4f} (+{diff:.4f} from best: {best_loss:.4f})")
      
      # Periodic save every 5 epochs
      if (epoch + 1) % 5 == 0:
            checkpoint = {
               "epoch": epoch,
               "model_state_dict": model.state_dict(),
               "optimizer_state_dict": optimizer.state_dict(),
               "loss": epoch_loss,
               "best_loss": best_loss,
               "threshold_same": th_same,
               "threshold_twin": th_twin
            }
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth"))
            log(f"Periodic save → epoch_{epoch+1}.pth")

   # ==================== FINAL REPORT ====================
   log("="*50)
   log("TRAINING COMPLETED", "SUCCESS")
   log(f"Epochs trained: {start_epoch+1} → {TOTAL_EPOCHS}")
   log(f"Best loss: {best_loss:.4f}")
   log(f"Final thresholds → SAME: {th_same:.4f} | TWIN: {th_twin:.4f}")
   log(f"Model saved: {CHECKPOINT_DIR}/siamese_best.pth")
   log(f"ONNX saved: {CHECKPOINT_DIR}/siamese_best.onnx")
   log("="*50)
   
   return model, best_loss, th_same, th_twin


if __name__ == "__main__":
   train()