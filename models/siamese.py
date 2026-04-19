import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import numpy as np
from config import IMG_SIZE


class SiameseNetwork(nn.Module):
   def __init__(self):
      super().__init__()

      base_model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
      )

      self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
      )

      self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64)
      )

   def forward_once(self, x):
      x = self.feature_extractor(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)

      # normalize embeddings
      x = F.normalize(x, p=2, dim=1)
      return x

   def forward(self, x1, x2):
      out1 = self.forward_once(x1)
      out2 = self.forward_once(x2)
      return out1, out2


class ContrastiveLoss(nn.Module):
   def __init__(self, margin=1.0):
      super().__init__()
      self.margin = margin

   def forward(self, out1, out2, label):
      distance = F.pairwise_distance(out1, out2)

      loss = torch.mean(
            label * torch.pow(distance, 2) +
            (1 - label) * torch.pow(
               torch.clamp(self.margin - distance, min=0.0), 2
            )
      )
      return loss


# 🔥 Mobile-friendly preprocessing
def preprocess_image(path):
   img = Image.open(path).convert("RGB")
   img = img.resize((IMG_SIZE, IMG_SIZE))

   img = np.array(img).astype("float32") / 255.0

   mean = np.array([0.485, 0.456, 0.406])
   std = np.array([0.229, 0.224, 0.225])

   img = (img - mean) / std
   img = np.transpose(img, (2, 0, 1))
   img = np.expand_dims(img, axis=0)

   return torch.tensor(img, dtype=torch.float32)


def compare_faces(model, img1, img2, device, th_same, th_twin):
   model.eval()

   img1 = preprocess_image(img1).to(device)
   img2 = preprocess_image(img2).to(device)

   with torch.no_grad():
      e1 = model.forward_once(img1)
      e2 = model.forward_once(img2)

      distance = F.pairwise_distance(e1, e2).item()

      if distance < th_same:
            result = "Same Person"
      elif distance < th_twin:
            result = "Twin"
      else:
            result = "Different People"

   return distance, result