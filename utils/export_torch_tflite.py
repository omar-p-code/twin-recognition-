import sys
import os
import torch
import onnx
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork  # your existing Siamese model class
from config import IMG_SIZE, DEVICE  # adjust as needed

CHECKPOINT_PATH = "checkpoints/checkpoint.pth"
OUTPUT_ONNX = "model.onnx"
OUTPUT_TFLITE = "model.tflite"

def load_model():
    """Load the trained SiameseNetwork model."""
    model = SiameseNetwork()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def export_onnx(model):
    """Export ONNX with two inputs and two outputs."""
    dummy_a = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    dummy_b = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    torch.onnx.export(
        model,
        (dummy_a, dummy_b),
        OUTPUT_ONNX,
        input_names=["image_a", "image_b"],
        output_names=["embedding_a", "embedding_b"],
        opset_version=12,
        dynamic_axes={
            "image_a": {0: "batch_size"},
            "image_b": {0: "batch_size"},
            "embedding_a": {0: "batch_size"},
            "embedding_b": {0: "batch_size"}
        }
    )
    # Verify ONNX
    onnx_model = onnx.load(OUTPUT_ONNX)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX exported with two inputs: image_a, image_b")
    return OUTPUT_ONNX

def convert_to_tflite(onnx_path):
    """Convert ONNX to TFLite while preserving two inputs."""
    # Run onnx2tf without unknown flags
    cmd = ["onnx2tf", "-i", onnx_path, "-o", "."]
    subprocess.run(cmd, check=True)

    # Find the generated .tflite file (it may be model_float32.tflite or model.tflite)
    tflite_files = [f for f in os.listdir(".") if f.endswith(".tflite")]
    if not tflite_files:
        raise FileNotFoundError("No .tflite file was generated.")
    # Use the first one found
    generated = tflite_files[0]
    os.rename(generated, OUTPUT_TFLITE)
    print(f"✅ TFLite exported: {OUTPUT_TFLITE} (two inputs preserved)")
    
def main():
    print("Loading Siamese model...")
    model = load_model()
    print("Exporting ONNX with two inputs...")
    export_onnx(model)
    print("Converting ONNX → TFLite with two inputs...")
    convert_to_tflite(OUTPUT_ONNX)
    print("🎉 Done! Your TFLite model now accepts two separate inputs.")

if __name__ == "__main__":
    main()