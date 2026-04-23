import torch
import torchvision.transforms as transforms

transform = transforms.Compose([
   transforms.Resize((128, 128)),
   transforms.ToTensor(),
])


def preprocess_image(img):
   """
   Converts PIL image to model-ready tensor
   """
   if img is None:
      return None

   try:
      img = transform(img)
      img = img.unsqueeze(0)  # add batch dimension -> [1, C, H, W]
      return img
   except Exception:
      return None