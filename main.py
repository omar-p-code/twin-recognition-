from inference.predict import load_model
from config import DEVICE
from models.siamese import compare_faces

if __name__ == "__main__":
   print("Starting the Twin Recognition project...")
   
   # Run training
   # train()
   
   # After training, you can test inference like this:
   model, th_same, th_twin = load_model("checkpoints/siamese_best.pth", DEVICE)
   distance, res = compare_faces(model, "data/val/Abdullah_Gul/Abdullah_Gul_0007.jpg", "data/val/Alejandro_Toledo/Alejandro_Toledo_0005.jpg", th_same, th_twin)
   print(f"Distance: {distance:.4f} → {res}")