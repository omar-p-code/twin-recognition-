from inference.predict import load_model
from config import DEVICE
from models.siamese import compare_faces
from training.train import train


train()  # Train the model before running inference



if __name__ == "__main__":
   print("Starting the Twin Recognition project...")
   
   # Load model
   checkpoint_path = "checkpoints/checkpoint.pth"
   
   # Check if model exists
   import os
   if not os.path.exists(checkpoint_path):
      print(f"❌ Model not found at: {checkpoint_path}")
      print("Please train the model first using: python train.py")
      exit(1)
   
   # Load model and thresholds
   model, th_same, th_twin = load_model(checkpoint_path, DEVICE)
   print(f"✓ Model loaded successfully")
   print(f"  Thresholds: same={th_same}, twin={th_twin}")
   
   # Image paths
   img1_path = "data/val/Abdullah_Gul/Abdullah_Gul_0007.jpg"
   img2_path = "data/val/Andy_Roddick/Andy_Roddick_0001.jpg"
   
   # Fix: Remove extra .jpg extension if present
   if img1_path.endswith('.jpg.jpg'):
      img1_path = img1_path.replace('.jpg.jpg', '.jpg')
   if img2_path.endswith('.jpg.jpg'):
      img2_path = img2_path.replace('.jpg.jpg', '.jpg')
   
   print(f"\n📸 Image 1: {img1_path}")
   print(f"📸 Image 2: {img2_path}")
   
   # Check if images exist
   if not os.path.exists(img1_path):
      print(f"❌ Image not found: {img1_path}")
      # Try to find alternative
      import glob
      files = glob.glob("data/val/*/*.jpg")
      if files:
            print(f"  Available images: {files[:3]}")
      exit(1)
   
   if not os.path.exists(img2_path):
      print(f"❌ Image not found: {img2_path}")
      exit(1)
   
   print("\n🔄 Comparing faces...")
   
   # FIX: compare_faces returns 3 values: (distance, result, detection_info)
   distance, result, detection_info = compare_faces(
      model, 
      img1_path, 
      img2_path, 
      th_same, 
      th_twin, 
      device=DEVICE, 
      detect_faces=True
   )
   
   # Check if comparison was successful
   if distance is None:
      print(f"\n❌ Comparison failed: {result}")
      if detection_info:
            print(f"  Face 1: {detection_info.get('face1_message', 'Unknown error')}")
            print(f"  Face 2: {detection_info.get('face2_message', 'Unknown error')}")
   else:
      print(f"\n{'='*40}")
      print(f"Thresholds: same={th_same}, twin={th_twin}")
      print(f"📊 RESULT:")
      print(f"  Distance: {distance:.4f}")
      print(f"  Result: {result}")
      
      if result == "Same Person":
            print(f"  ✅ These are the SAME person")
      elif result == "Twins":
            print(f"  👯 These appear to be TWINS")
      else:
            print(f"  ❌ These are DIFFERENT people")
      print(f"{'='*40}")
      
      
      

# from ui.app import TwinApp

# if __name__ == "__main__":
#    print("Starting the Twin Recognition App...")
#    TwinApp().run()