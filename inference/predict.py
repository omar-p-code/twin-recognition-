import torch
from models.siamese import SiameseNetwork, compare_faces
from config import DEVICE, THRESHOLD

def load_model(checkpoint_path):
   model = SiameseNetwork().to(DEVICE)
   model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
   model.eval()
   return model

# Example usage
if __name__ == "__main__":
   model = load_model("checkpoints/siamese_best.pth")
   
   img1_path = "data/test/personA.jpg"
   img2_path = "data/test/personB.jpg"
   
   distance, val = compare_faces(model, img1_path, img2_path, device=DEVICE, threshold_same=THRESHOLD)
   
   print(f"Distance: {distance:.4f}")
   print(f"Same person (twin)? → {val}")