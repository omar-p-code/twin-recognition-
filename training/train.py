# training/train.py - Smart training with default thresholds and continuation
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import glob
import json

from models.siamese import SiameseNetwork, ContrastiveLoss
from utils.dataset import PairDataset
from config import *
from utils.export_onnx import export_onnx

# Default thresholds (proven to work well)
DEFAULT_TH_SAME = 0.4
DEFAULT_TH_TWIN = 0.7

# Training state tracking
class TrainingState:
   def __init__(self, checkpoint_dir="checkpoints"):
      self.checkpoint_dir = checkpoint_dir
      self.state_file = os.path.join(checkpoint_dir, "training_state.json")
      self.load()
   
   def load(self):
      if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
               data = json.load(f)
               self.best_loss = data.get('best_loss', float('inf'))
               self.stable_epochs = data.get('stable_epochs', 0)
               self.need_calibration = data.get('need_calibration', False)
      else:
            self.best_loss = float('inf')
            self.stable_epochs = 0
            self.need_calibration = False
   
   def save(self):
      with open(self.state_file, 'w') as f:
            json.dump({
               'best_loss': self.best_loss,
               'stable_epochs': self.stable_epochs,
               'need_calibration': self.need_calibration
            }, f)
   
   def update(self, epoch_loss):
      if epoch_loss < self.best_loss:
            self.best_loss = epoch_loss
            self.stable_epochs = 0
      else:
            self.stable_epochs += 1
      
      # Need calibration if loss hasn't improved for 5 epochs
      self.need_calibration = self.stable_epochs >= 5
      self.save()

def find_latest_checkpoint(checkpoint_dir):
   """Find the latest checkpoint in the directory"""
   checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
   
   if not checkpoint_files:
      return None
   
   def get_epoch_num(path):
      try:
            basename = os.path.basename(path)
            epoch_str = basename.split("epoch_")[1].split(".")[0]
            return int(epoch_str)
      except:
            return 0
   
   checkpoint_files.sort(key=get_epoch_num, reverse=True)
   return checkpoint_files[0]

def load_checkpoint(model, optimizer, checkpoint_path, device):
   """Load checkpoint and return metadata"""
   if not os.path.exists(checkpoint_path):
      print("No checkpoint found. Starting from scratch.")
      return 0, float("inf"), DEFAULT_TH_SAME, DEFAULT_TH_TWIN
   
   print(f"📂 Loading checkpoint: {checkpoint_path}")
   checkpoint = torch.load(checkpoint_path, map_location=device)
   
   # Load model state
   if "model_state_dict" in checkpoint:
      model.load_state_dict(checkpoint["model_state_dict"])
   else:
      model.load_state_dict(checkpoint)
   
   # Load optimizer state if available
   if optimizer and "optimizer_state_dict" in checkpoint:
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      print("  ✓ Optimizer state restored")
   
   # Get training state
   start_epoch = checkpoint.get("epoch", 0)
   if start_epoch > 0:
      start_epoch += 1
   
   best_loss = checkpoint.get("best_loss", float("inf"))
   
   # Use stored thresholds or defaults
   th_same = checkpoint.get("threshold_same", DEFAULT_TH_SAME)
   th_twin = checkpoint.get("threshold_twin", DEFAULT_TH_TWIN)
   
   print(f"  ✓ Resuming from epoch {start_epoch}")
   print(f"  ✓ Best loss: {best_loss:.4f}")
   print(f"  ✓ Current thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   
   return start_epoch, best_loss, th_same, th_twin

def calculate_thresholds_if_needed(model, loader, device, current_th_same, current_th_twin, force=False):
   """
   Calculate new thresholds only if needed or forced
   Returns: (new_th_same, new_th_twin, did_calibrate)
   """
   # Don't calibrate for first 10 epochs (model too unstable)
   global current_epoch
   if current_epoch < 10 and not force:
      print(f"  ℹ️ Skipping calibration (epoch {current_epoch+1} < 10)")
      return current_th_same, current_th_twin, False
   
   if not force:
      print(f"  ℹ️ Using default thresholds (no calibration needed)")
      return current_th_same, current_th_twin, False
   
   print("  🔧 Calibrating thresholds...")
   model.eval()
   same_dist = []
   diff_dist = []
   
   with torch.no_grad():
      for img1, img2, label in tqdm(loader, desc="  Calculating distances"):
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
   
   if len(same_dist) > 0:
      th_same = np.percentile(same_dist, 90)
   else:
      th_same = DEFAULT_TH_SAME
   
   if len(diff_dist) > 0:
      th_twin = np.percentile(diff_dist, 10)
   else:
      th_twin = DEFAULT_TH_TWIN
   
   model.train()
   
   print(f"  ✓ New thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   
   # Compare with defaults
   if abs(th_same - DEFAULT_TH_SAME) > 0.1 or abs(th_twin - DEFAULT_TH_TWIN) > 0.1:
      print(f"  ⚠️ Thresholds changed significantly from defaults!")
      print(f"     Default: same={DEFAULT_TH_SAME}, twin={DEFAULT_TH_TWIN}")
   
   return th_same, th_twin, True

def train(extra_epochs=0, force_recalibrate=False):
   """
   Train the model
   Args:
      extra_epochs: Number of additional epochs to train beyond NUM_EPOCHS
      force_recalibrate: Force threshold recalibration even if not needed
   """
   os.makedirs(CHECKPOINT_DIR, exist_ok=True)
   
   # Calculate total epochs
   total_epochs = NUM_EPOCHS + extra_epochs
   
   # Data transforms
   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.RandomHorizontalFlip(p=0.5),
      transforms.RandomRotation(5),
      transforms.ToTensor(),
      transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
      )
   ])
   
   print("📁 Loading dataset...")
   dataset = PairDataset(f"{DATA_DIR}/train", transform)
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
   print(f"✓ Dataset loaded: {len(dataset)} pairs")
   print(f"✓ Total epochs to train: {total_epochs}")
   
   # Initialize model
   model = SiameseNetwork().to(DEVICE)
   criterion = ContrastiveLoss(margin=MARGIN)
   optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode='min', factor=0.5, patience=3
   )
   
   # Load from checkpoint if exists
   start_epoch = 0
   best_loss = float("inf")
   th_same = DEFAULT_TH_SAME
   th_twin = DEFAULT_TH_TWIN
   
   latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR)
   if latest_checkpoint:
      start_epoch, best_loss, th_same, th_twin = load_checkpoint(
            model, optimizer, latest_checkpoint, DEVICE
      )
   else:
      print("📂 No checkpoint found. Starting fresh training.")
      print(f"📂 Using default thresholds: same={DEFAULT_TH_SAME}, twin={DEFAULT_TH_TWIN}")
   
   # Training state tracker
   training_state = TrainingState()
   
   print(f"\n{'='*60}")
   print(f"🚀 Starting Training")
   print(f"   Device: {DEVICE}")
   print(f"   Total epochs: {total_epochs}")
   print(f"   Start epoch: {start_epoch + 1}")
   print(f"   Batch size: {BATCH_SIZE}")
   print(f"   Learning rate: {LEARNING_RATE}")
   print(f"   Margin: {MARGIN}")
   print(f"   Default thresholds: same={DEFAULT_TH_SAME}, twin={DEFAULT_TH_TWIN}")
   print(f"{'='*60}\n")
   
   global current_epoch
   for epoch in range(start_epoch, total_epochs):
      current_epoch = epoch
      model.train()
      total_loss = 0
      num_batches = 0
      
      print(f"\n📊 Epoch {epoch+1}/{total_epochs}")
      print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
      print(f"   Current thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
      
      progress_bar = tqdm(loader, desc=f"Training")
      for img1, img2, label in progress_bar:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            label = label.to(DEVICE)
            
            optimizer.zero_grad()
            out1, out2 = model(img1, img2)
            loss = criterion(out1, out2, label)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
               'loss': f'{loss.item():.4f}',
               'avg_loss': f'{total_loss/num_batches:.4f}'
            })
      
      epoch_loss = total_loss / len(loader)
      print(f"\n  📈 Epoch {epoch+1} Average Loss: {epoch_loss:.4f}")
      
      # Update training state
      training_state.update(epoch_loss)
      
      # Update learning rate
      scheduler.step(epoch_loss)
      
      # Smart threshold calibration - only when needed
      should_calibrate = force_recalibrate or training_state.need_calibration
      th_same, th_twin, did_calibrate = calculate_thresholds_if_needed(
            model, loader, DEVICE, th_same, th_twin, force=should_calibrate
      )
      
      if did_calibrate:
            print(f"  ✓ Thresholds updated due to stable loss")
            training_state.need_calibration = False
            training_state.save()
      
      # Save checkpoint
      checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "best_loss": min(best_loss, epoch_loss),
            "threshold_same": th_same,
            "threshold_twin": th_twin,
            "default_th_same": DEFAULT_TH_SAME,
            "default_th_twin": DEFAULT_TH_TWIN
      }
      
      # Save regular checkpoint
      torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth"))
      
      # Save best model
      if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_best.pth"))
            print(f"  ⭐ New best model! (loss: {epoch_loss:.4f})")
      
      # Save latest model
      torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "siamese_latest.pth"))
      
      # Export to ONNX every 20 epochs
      if (epoch + 1) % 20 == 0:
            try:
               export_onnx(model, os.path.join(CHECKPOINT_DIR, f"siamese_epoch_{epoch+1}.onnx"))
               print(f"  ✓ ONNX exported")
            except Exception as e:
               print(f"  ⚠️ ONNX export failed: {e}")
      
      print(f"  💾 Checkpoint saved")
      print(f"  📊 Best loss: {best_loss:.4f}")
      
      # Early stopping if loss is very low and stable
      if epoch_loss < 0.01 and training_state.stable_epochs > 10:
            print(f"\n🎉 Training converged! Loss is very low and stable.")
            print(f"   Stopping early at epoch {epoch+1}")
            break
   
   print(f"\n{'='*60}")
   print("✅ Training completed successfully!")
   print(f"   Best loss: {best_loss:.4f}")
   print(f"   Final thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
   print(f"   Default thresholds: same={DEFAULT_TH_SAME}, twin={DEFAULT_TH_TWIN}")
   print(f"{'='*60}")

def continue_training(extra_epochs=20):
   """Continue training for additional epochs"""
   print(f"\n{'='*60}")
   print(f"🔄 Continuing training for {extra_epochs} additional epochs")
   print(f"{'='*60}\n")
   train(extra_epochs=extra_epochs, force_recalibrate=False)

def continue_with_recalibration(extra_epochs=20):
   """Continue training AND force threshold recalibration"""
   print(f"\n{'='*60}")
   print(f"🔄 Continuing training with forced recalibration for {extra_epochs} epochs")
   print(f"{'='*60}\n")
   train(extra_epochs=extra_epochs, force_recalibrate=True)

if __name__ == "__main__":
   import argparse
   
   parser = argparse.ArgumentParser(description='Train Siamese Network')
   parser.add_argument('--fresh', action='store_true', help='Start fresh training')
   parser.add_argument('--continue', dest='continue_train', action='store_true', 
                     help='Continue training from last checkpoint')
   parser.add_argument('--extra', type=int, default=0, 
                     help='Extra epochs to train beyond NUM_EPOCHS')
   parser.add_argument('--recalibrate', action='store_true',
                     help='Force threshold recalibration')
   
   args = parser.parse_args()
   
   if args.fresh:
      # Delete existing checkpoints
      import shutil
      if os.path.exists(CHECKPOINT_DIR):
            shutil.rmtree(CHECKPOINT_DIR)
      os.makedirs(CHECKPOINT_DIR, exist_ok=True)
      print("✅ Starting fresh training")
      train(extra_epochs=args.extra, force_recalibrate=args.recalibrate)
   elif args.continue_train:
      continue_training(extra_epochs=args.extra if args.extra > 0 else 20)
   else:
      train(extra_epochs=args.extra, force_recalibrate=args.recalibrate)

# example usage:

"""
   # 1. بدء التدريب (سيستأنف تلقائياً)
python training/train.py

# 2. بدء تدريب جديد من الصفر
python training/train.py --fresh

# 3. عرض جميع checkpoints المتاحة
python inference/predict.py

# 4. تحميل آخر checkpoint
python -c "from inference.predict import load_latest_checkpoint; load_latest_checkpoint()"

# 5. استئناف التدريب ببساطة
python resume_training.py


"""