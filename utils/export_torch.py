import sys
import torch
import onnx
import os
import tensorflow as tf
import tf2onnx

sys.path.append(os.path.dirname(__file__))

CHECKPOINT_DIR = "checkpoints/"
INPUT_SIZE = (1, 3, 224, 224)

def load_model():
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        model = checkpoint
        model.eval()
        return model
    
    from torchvision import models
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 256)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def export_onnx(model):
    dummy_input = torch.randn(*INPUT_SIZE)
    onnx_path = "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX exported: {onnx_path}")
    return onnx_path

def convert_to_tflite_from_onnx(onnx_path):
    """Convert ONNX directly to TFLite using tf2onnx"""
    # Load ONNX and convert to TensorFlow
    tf_model_path = "tf_model"
    
    # Use tf2onnx's converter (it handles the conversion)
    import subprocess
    cmd = [
        "python", "-m", "tf2onnx.convert",
        "--onnx", onnx_path,
        "--output", f"{tf_model_path}/model.pb"
    ]
    subprocess.run(cmd, check=True)
    
    # Now convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    out_path = "model.tflite"
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite exported: {out_path}")
    return out_path

def main():
    print("Loading PyTorch model...")
    model = load_model()
    print("Exporting ONNX...")
    onnx_path = export_onnx(model)
    print("Converting ONNX → TFLite...")
    convert_to_tflite_from_onnx(onnx_path)
    print("🎉 Done!")

if __name__ == "__main__":
    main()