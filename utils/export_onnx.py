import torch
from config import DEVICE, IMG_SIZE

def export_onnx(model, path="siamese.onnx"):
   model.eval()

   dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

   torch.onnx.export(
      model.forward_once,   # important
      dummy_input,
      path,
      input_names=["input"],
      output_names=["embedding"],
      opset_version=11
   )

   print("ONNX exported successfully!")