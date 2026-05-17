"""
Compare two face images using the exported TFLite model (or PyTorch checkpoint).
Usage: python compare_faces_tflite.py <image1> <image2>
"""

import sys
import numpy as np
from PIL import Image
import argparse

# Try to import the appropriate TFLite runtime
try:
    from ai_edge_tflite.interpreter import Interpreter
    print("Using ai-edge-tflite")
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
        print("Using tflite-runtime")
    except ImportError:
        print("Neither ai-edge-tflite nor tflite-runtime found. Falling back to PyTorch.")
        Interpreter = None

# Constants (must match training)
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TH_SAME = 0.6227   # update from your last calibration
TH_TWIN = 0.6461   # update from your last calibration

def preprocess_image(image_path):
    """Load and preprocess an image to NHWC tensor (1,224,224,3) as float32."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0
    # Normalize
    for c in range(3):
        img_np[:,:,c] = (img_np[:,:,c] - MEAN[c]) / STD[c]
    # Add batch dimension: (224,224,3) -> (1,224,224,3)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np.astype(np.float32)

def compare_tflite(model_path, img1_path, img2_path):
    """Load TFLite model and compare two images."""
    if Interpreter is None:
        raise ImportError("No TFLite runtime available")
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # Preprocess images
    input1 = preprocess_image(img1_path)
    input2 = preprocess_image(img2_path)
    # Set inputs (two input tensors)
    interpreter.set_tensor(input_details[0]['index'], input1)
    interpreter.set_tensor(input_details[1]['index'], input2)
    interpreter.invoke()
    # Get outputs (two embeddings)
    emb1 = interpreter.get_tensor(output_details[0]['index'])[0]  # shape (256,)
    emb2 = interpreter.get_tensor(output_details[1]['index'])[0]
    # Compute Euclidean distance
    distance = np.linalg.norm(emb1 - emb2)
    return distance, emb1, emb2

def compare_pytorch(checkpoint_path, img1_path, img2_path, device='cpu'):
    """Fallback: compare using PyTorch model."""
    import torch
    import torchvision.transforms as transforms
    from models.siamese import SiameseNetwork
    # Load model
    model = SiameseNetwork()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.to(device)
    model.eval()
    # Transform
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(device)
    img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1, emb2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(emb1, emb2).item()
    return distance, emb1.cpu().numpy()[0], emb2.cpu().numpy()[0]

def classify(distance, th_same, th_twin):
    if distance < th_same:
        return "Same Person"
    elif distance < th_twin:
        return "Twins"
    else:
        return "Different People"

def main():
    parser = argparse.ArgumentParser(description="Compare two face images using trained model")
    parser.add_argument("img1", help="Path to first image")
    parser.add_argument("img2", help="Path to second image")
    parser.add_argument("--model", default="checkpoints/model.tflite",
                        help="Path to TFLite model (default: checkpoints/model.tflite)")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pth",
                        help="Path to PyTorch checkpoint (fallback)")
    parser.add_argument("--tflite", action="store_true", default=True,
                        help="Use TFLite model (default)")
    parser.add_argument("--pytorch", action="store_true",
                        help="Use PyTorch model instead of TFLite")
    parser.add_argument("--th_same", type=float, default=TH_SAME,
                        help="Threshold for same person")
    parser.add_argument("--th_twin", type=float, default=TH_TWIN,
                        help="Threshold for twins")
    args = parser.parse_args()

    try:
        if args.pytorch:
            print("Using PyTorch model...")
            distance, _, _ = compare_pytorch(args.checkpoint, args.img1, args.img2)
        else:
            print(f"Using TFLite model: {args.model}")
            distance, _, _ = compare_tflite(args.model, args.img1, args.img2)
        label = classify(distance, args.th_same, args.th_twin)
        print(f"Distance: {distance:.4f}")
        print(f"Result: {label}")
        print(f"Thresholds: same={args.th_same}, twin={args.th_twin}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()