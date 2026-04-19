import torch
from models.siamese import SiameseNetwork

DEVICE = "cpu"

model = SiameseNetwork().to(DEVICE)

checkpoint = torch.load(
   "checkpoints/siamese_best.pth",
   map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"], strict=False)
model.eval()

dummy = torch.randn(1, 3, 224, 224)

torch.onnx.export(
   model.forward_once,
   dummy,
   "siamese.onnx",
   input_names=["input"],
   output_names=["embedding"],
   opset_version=11
)

print("ONNX exported successfully")