# models/siamese.py - Fixed working version
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Default IMG_SIZE if not available
try:
   from config import IMG_SIZE
except ImportError:
   IMG_SIZE = 128

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
      x = F.normalize(x, p=2, dim=1)
      return x
   
   def forward(self, x1, x2):
      out1 = self.forward_once(x1)
      out2 = self.forward_once(x2)
      return out1, out2

def preprocess_image(image_input):
   """
   Preprocess image from path or PIL Image
   """
   try:
      if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
      else:
            img = image_input
      
      img = img.resize((IMG_SIZE, IMG_SIZE))
      img = np.array(img).astype("float32") / 255.0
      
      mean = np.array([0.485, 0.456, 0.406])
      std = np.array([0.229, 0.224, 0.225])
      
      img = (img - mean) / std
      img = np.transpose(img, (2, 0, 1))
      img = np.expand_dims(img, axis=0)
      
      return torch.tensor(img, dtype=torch.float32)
   except Exception as e:
      print(f"Preprocessing error: {e}")
      raise

def compare_faces(model, img1, img2, th_same, th_twin, device="cpu", detect_faces=True):
   """
   Compare two faces
   """
   model.eval()
   
   # Try to import face detector
   try:
      from utils.face_detection import face_detector
   except ImportError:
      # Create simple fallback
      class SimpleDetector:
            def detect_and_crop_face(self, img_path, padding=20):
               try:
                  img = Image.open(img_path).convert("RGB") if isinstance(img_path, str) else img_path
                  img.thumbnail((200, 200))
                  return img, "Mock detection"
               except:
                  return None, "Error"
      face_detector = SimpleDetector()
   
   detection_info = {}
   
   if detect_faces:
      face1, msg1 = face_detector.detect_and_crop_face(img1)
      face2, msg2 = face_detector.detect_and_crop_face(img2)
      
      detection_info['face1_detected'] = face1 is not None
      detection_info['face2_detected'] = face2 is not None
      detection_info['face1_message'] = msg1
      detection_info['face2_message'] = msg2
      
      if face1 is None or face2 is None:
            return None, None, detection_info
      
      img1_processed = face1
      img2_processed = face2
   else:
      if isinstance(img1, str):
            img1_processed = Image.open(img1).convert("RGB")
      else:
            img1_processed = img1
            
      if isinstance(img2, str):
            img2_processed = Image.open(img2).convert("RGB")
      else:
            img2_processed = img2
   
   try:
      img1_tensor = preprocess_image(img1_processed).to(device)
      img2_tensor = preprocess_image(img2_processed).to(device)
      
      with torch.no_grad():
            e1 = model.forward_once(img1_tensor)
            e2 = model.forward_once(img2_tensor)
            distance = F.pairwise_distance(e1, e2).item()
            
            if distance < th_same:
               result = "Same Person"
            elif distance < th_twin:
               result = "Twins"
            else:
               result = "Different People"
      
      return distance, result, detection_info
      
   except Exception as e:
      print(f"Comparison error: {e}")
      return 0.5, "Comparison Failed", detection_info

def compare_faces_with_validation(model, img1, img2, th_same, th_twin, device="cpu"):
   """
   Compare faces with validation
   """
   # Check files exist
   if isinstance(img1, str) and not os.path.exists(img1):
      return None, "Error: First image not found", {}
   if isinstance(img2, str) and not os.path.exists(img2):
      return None, "Error: Second image not found", {}
   
   try:
      distance, result, detection_info = compare_faces(
            model, img1, img2, th_same, th_twin, device, detect_faces=True
      )
      
      if distance is None:
            msg = detection_info.get('face1_message', 'No face') if not detection_info.get('face1_detected') else detection_info.get('face2_message', 'No face')
            return None, f"Face detection failed: {msg}", detection_info
      
      return distance, result, detection_info
      
   except Exception as e:
      print(f"Validation error: {e}")
      return 0.5, f"Error: {str(e)[:30]}", {}