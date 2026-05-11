"""
ONNX Runtime inference — used on Android instead of PyTorch.
Export the model first with: python main.py --train
(or call training.train.export_onnx() directly)
"""

import numpy as np
import os
from PIL import Image

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    print("! onnxruntime not installed — ONNX inference unavailable")

IMG_SIZE       = 224
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(img_path) -> np.ndarray:
    """Load and preprocess a face image into a (1, 3, H, W) float32 array."""
    img = Image.open(img_path).convert("RGB") if isinstance(img_path, str) else img_path.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0           # [H, W, 3]
    arr = (arr - NORMALIZE_MEAN) / NORMALIZE_STD             # normalise
    arr = arr.transpose(2, 0, 1)[np.newaxis, :]              # [1, 3, H, W]
    return arr


class ONNXPredictor:
    """
    Drop-in ONNX replacement for the PyTorch SiameseNetwork.
    Used on Android where PyTorch is unavailable.
    """

    def __init__(self, model_path: str):
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is not installed.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Use CPU provider — fine for mobile
        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self.input1_name = self.session.get_inputs()[0].name
        self.input2_name = self.session.get_inputs()[1].name
        print(f"✓ ONNX model loaded: {model_path}")

    def compare(self, img1_path, img2_path, th_same: float, th_twin: float):
        """
        Compare two face images.
        Returns (distance: float, result: str)
        """
        t1 = _preprocess(img1_path)
        t2 = _preprocess(img2_path)

        emb1, emb2 = self.session.run(
            None, {self.input1_name: t1, self.input2_name: t2}
        )

        # L2 distance between embeddings
        diff     = emb1[0] - emb2[0]
        distance = float(np.sqrt(np.sum(diff ** 2)))

        if distance < th_same:
            result = "Same Person"
        elif distance < th_twin:
            result = "Twins"
        else:
            result = "Different People"

        return distance, result


def load_onnx_model(checkpoint_dir: str = "checkpoints"):
    """
    Load the ONNX model and thresholds saved alongside it.
    Falls back to default thresholds if the .pth file is missing.
    """
    onnx_path = os.path.join(checkpoint_dir, "model.onnx")
    pth_path  = os.path.join(checkpoint_dir, "checkpoint.pth")

    th_same, th_twin = 0.35, 0.60

    if os.path.exists(pth_path):
        try:
            import torch
            ckpt    = torch.load(pth_path, map_location="cpu")
            th_same = ckpt.get("threshold_same", th_same)
            th_twin = ckpt.get("threshold_twin", th_twin)
        except Exception as e:
            print(f"! Could not read thresholds from checkpoint: {e}")

    predictor = ONNXPredictor(onnx_path)
    return predictor, th_same, th_twin
