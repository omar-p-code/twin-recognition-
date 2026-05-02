import torchvision.transforms as transforms
from config import *

transform = transforms.Compose([
   transforms.Resize((IMG_SIZE, IMG_SIZE)),
   transforms.ToTensor(),
   transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
])


def preprocess_image(img):
   """
   Converts PIL image to model-ready tensor
   """

   if img is None:
      return None

   try:
      img = img.convert("RGB")

      img = transform(img)

      # [C,H,W] -> [1,C,H,W]
      img = img.unsqueeze(0)

      return img

   except Exception as e:
      print(f"Preprocess error: {e}")
      return None