import torch
from models.siamese import SiameseNetwork


def load_model(path, device):
   model = SiameseNetwork().to(device)

   checkpoint = torch.load(path, map_location=device)

   if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
      state_dict = checkpoint["model_state_dict"]

   elif isinstance(checkpoint, dict):
      state_dict = checkpoint

   else:
      raise ValueError("Invalid checkpoint format")

   model.load_state_dict(state_dict, strict=False)
   model.eval()

   th_same = checkpoint.get("threshold_same", 0.4)
   th_twin = checkpoint.get("threshold_twin", 0.7)

   return model, th_same, th_twin