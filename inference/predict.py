# inference/predict.py - Updated to read all checkpoint info
import torch
import os
from models.siamese import SiameseNetwork

def load_model(path, device):
   """Load model from checkpoint with complete info"""
   model = SiameseNetwork().to(device)
   
   if not os.path.exists(path):
      print(f"❌ Model not found: {path}")
      return model, 0.4, 0.7
   
   print(f"📂 Loading model from: {path}")
   checkpoint = torch.load(path, map_location=device)
   
   # Handle different checkpoint formats
   if isinstance(checkpoint, dict):
      # Check for model_state_dict
      if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print("  ✓ Model weights loaded")
      else:
            model.load_state_dict(checkpoint)
            print("  ✓ Model weights loaded (direct format)")
      
      # Get thresholds
      th_same = checkpoint.get("threshold_same", 0.4)
      th_twin = checkpoint.get("threshold_twin", 0.7)
      
      # Get training info if available
      epoch = checkpoint.get("epoch", -1)
      loss = checkpoint.get("loss", -1)
      best_loss = checkpoint.get("best_loss", -1)
      
      if epoch >= 0:
            print(f"  ✓ Training info: epoch {epoch+1}, loss: {loss:.4f}")
      if best_loss >= 0:
            print(f"  ✓ Best loss: {best_loss:.4f}")
      
      print(f"  ✓ Thresholds: same={th_same:.4f}, twin={th_twin:.4f}")
      
   else:
      print(f"  ⚠️ Unknown checkpoint format")
      th_same = 0.4
      th_twin = 0.7
   
   model.eval()
   return model, th_same, th_twin

def load_latest_checkpoint(checkpoint_dir="checkpoints", device="cpu"):
   """Automatically load the latest checkpoint"""
   import glob
   
   # Find all checkpoints
   checkpoints = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
   
   if not checkpoints:
      # Try best model
      best_path = os.path.join(checkpoint_dir, "siamese_best.pth")
      if os.path.exists(best_path):
            return load_model(best_path, device)
      else:
            print(f"❌ No checkpoints found in {checkpoint_dir}")
            return None, None, None
   
   # Get the latest epoch
   def get_epoch(path):
      try:
            basename = os.path.basename(path)
            epoch_str = basename.split("epoch_")[1].split(".")[0]
            return int(epoch_str)
      except:
            return 0
   
   latest = max(checkpoints, key=get_epoch)
   print(f"📂 Loading latest checkpoint: {os.path.basename(latest)}")
   
   return load_model(latest, device)

def get_checkpoint_info(checkpoint_path):
   """Extract information from checkpoint without loading the model"""
   if not os.path.exists(checkpoint_path):
      return None
   
   checkpoint = torch.load(checkpoint_path, map_location="cpu")
   
   info = {
      "path": checkpoint_path,
      "size_mb": os.path.getsize(checkpoint_path) / (1024 * 1024),
   }
   
   if isinstance(checkpoint, dict):
      info["epoch"] = checkpoint.get("epoch", -1)
      info["loss"] = checkpoint.get("loss", -1)
      info["best_loss"] = checkpoint.get("best_loss", -1)
      info["th_same"] = checkpoint.get("threshold_same", 0.4)
      info["th_twin"] = checkpoint.get("threshold_twin", 0.7)
   
   return info

def list_all_checkpoints(checkpoint_dir="checkpoints"):
   """List all available checkpoints with information"""
   import glob
   
   checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
   
   if not checkpoints:
      print("No checkpoints found")
      return
   
   print("\n" + "="*60)
   print("Available Checkpoints")
   print("="*60)
   
   # Sort by modification time
   checkpoints.sort(key=os.path.getmtime, reverse=True)
   
   for ckpt in checkpoints:
      name = os.path.basename(ckpt)
      size_mb = os.path.getsize(ckpt) / (1024 * 1024)
      
      # Try to get info
      try:
            checkpoint = torch.load(ckpt, map_location="cpu")
            if isinstance(checkpoint, dict):
               epoch = checkpoint.get("epoch", -1)
               loss = checkpoint.get("loss", -1)
               th_same = checkpoint.get("threshold_same", 0.4)
               th_twin = checkpoint.get("threshold_twin", 0.7)
               
               print(f"\n📁 {name} ({size_mb:.1f} MB)")
               if epoch >= 0:
                  print(f"   Epoch: {epoch+1}")
               if loss >= 0:
                  print(f"   Loss: {loss:.4f}")
               print(f"   Thresholds: same={th_same:.3f}, twin={th_twin:.3f}")
            else:
               print(f"\n📁 {name} ({size_mb:.1f} MB) - Unknown format")
      except:
            print(f"\n📁 {name} ({size_mb:.1f} MB) - Corrupted?")
   
   print("\n" + "="*60)

if __name__ == "__main__":
   # Test the loader
   import sys
   if len(sys.argv) > 1:
      model, th_same, th_twin = load_model(sys.argv[1], "cpu")
      print(f"\n✅ Model loaded successfully")
   else:
      list_all_checkpoints()