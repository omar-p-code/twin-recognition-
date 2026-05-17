import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork
from config import CHECKPOINT_DIR, IMG_SIZE, DEVICE

def export_to_tflite():
    # 1. Load model
    model = SiameseNetwork().to(DEVICE)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Model loaded and set to eval mode")

    # 2. Export to ONNX
    onnx_path = os.path.join(CHECKPOINT_DIR, "model.onnx")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    torch.onnx.export(
        model,
        (dummy_input, dummy_input),
        onnx_path,
        input_names=["img1", "img2"],
        output_names=["emb1", "emb2"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "img1": {0: "batch_size"},
            "img2": {0: "batch_size"},
            "emb1": {0: "batch_size"},
            "emb2": {0: "batch_size"}
        }
    )
    print(f"✅ ONNX model saved to {onnx_path}")

    # 3. Convert ONNX to TFLite using only essential parameters
    from onnx2tf import convert

    output_dir = os.path.join(CHECKPOINT_DIR, "tflite_model")
    os.makedirs(output_dir, exist_ok=True)

    # Use minimal, widely supported parameters
    convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        output_integer_quantized_tflite=True,   # enable integer quantization
        not_use_onnxsim=True,
        keep_input_and_output_names=True,
    )
    print(f"✅ TFLite models saved to {output_dir}")

if __name__ == "__main__":
    export_to_tflite()