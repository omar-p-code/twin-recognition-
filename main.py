from training.train import train
from inference.predict import load_model
from config import DEVICE
from models.siamese import compare_faces

if __name__ == "__main__":
   print("Starting the Twin Recognition project...")
   
   # Run training
   train()
   
   # After training, you can test inference like this:
   model = load_model("checkpoints/siamese_best.pth")
   distance, res = compare_faces(model, f"data/val/Abdullah_Gul/Abdullah_Gul_0007.jpg", f"data/val/Alejandro_Toledo/Alejandro_Toledo_0005.jpg", device=DEVICE)
   print(f"Distance: {distance:.4f} → {res}")