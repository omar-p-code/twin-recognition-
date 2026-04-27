import onnxruntime as ort
# import numpy as np
from PIL import Image
import torchvision.transforms as transforms

session = ort.InferenceSession("siamese.onnx")

transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]
   )
])


def get_embedding(img_path):
   img = Image.open(img_path).convert("RGB")
   img = transform(img).unsqueeze(0).numpy()

   return session.run(None, {"input": img})[0]