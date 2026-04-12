import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from config import IMG_SIZE


class SiameseNetwork(nn.Module):
   def __init__(self):
      super(SiameseNetwork, self).__init__()

      base_model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
      )

      self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
      )

      self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128)
      )

   def forward_once(self, x):
      x = self.feature_extractor(x)
      x = x.view(x.size(0), -1)
      x = self.fc(x)

      # Normalize the output to unit length
      x = F.normalize(x, p=2, dim=1)

      return x

   def forward(self, anchor, positive, negative):
      anchor_emb = self.forward_once(anchor)
      positive_emb = self.forward_once(positive)
      negative_emb = self.forward_once(negative)

      return anchor_emb, positive_emb, negative_emb


class TripletLoss(nn.Module):
   def __init__(self, margin=1.5):
      super().__init__()
      self.loss_fn = nn.TripletMarginLoss(
            margin=margin,
            p=2
      )

   def forward(self, anchor, positive, negative):
      return self.loss_fn(anchor, positive, negative)


def compare_faces(
   model,
   img1_path,
   img2_path,
   device,
   threshold_same=0.5,
   threshold_twin=1.0
):
   model.eval()

   transform = transforms.Compose([
      transforms.Resize((IMG_SIZE, IMG_SIZE)),
      transforms.ToTensor(),
      transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
      )
   ])

   img1 = transform(
      Image.open(img1_path).convert("RGB")
   ).unsqueeze(0).to(device)

   img2 = transform(
      Image.open(img2_path).convert("RGB")
   ).unsqueeze(0).to(device)

   with torch.no_grad():
      out1 = model.forward_once(img1)
      out2 = model.forward_once(img2)

      distance = torch.nn.functional.pairwise_distance(
            out1,
            out2
      ).item()

      if distance < threshold_same:
            result = "Same Person"
      elif distance < threshold_twin:
            result = "Possible Twin"
      else:
            result = "Different People"

   return distance, result
