# export_for_mobile.py
import torch
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEVICE, IMG_SIZE

# Import your model architecture (adjust based on your actual model file)
# Option A: If you have a siamese.py file
try:
    from models.siamese import SiameseNetwork
    print("✅ Imported SiameseNetwork from models.siamese")
except:
    try:
        from models.siamese import SiameseNetwork
        # Option B: If your model is defined elsewhere
        print("✅ Imported SiameseNetwork from model")
    except:
        # Option C: Define your model architecture here (if needed)
        print("⚠️  Could not import model. Please specify your model class.")
        print("Define your SiameseNetwork class or update the import.")
        exit(1)

def export_full_model(model, thresholds, onnx_path="model.onnx", thresholds_path="thresholds.json"):
    """Export both ONNX model and thresholds for mobile."""
    model.eval()
    model.to(DEVICE)
    
    print(f"📦 Exporting ONNX model to {onnx_path}...")
    
    # Create dummy inputs for both branches
    dummy_input_a = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    dummy_input_b = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Export with two inputs and two outputs
    torch.onnx.export(
        model,
        (dummy_input_a, dummy_input_b),
        onnx_path,
        input_names=["image_a", "image_b"],
        output_names=["embedding_a", "embedding_b"],
        dynamic_axes={
            "image_a": {0: "batch_size"},
            "image_b": {0: "batch_size"},
            "embedding_a": {0: "batch_size"},
            "embedding_b": {0: "batch_size"}
        },
        opset_version=11,
        do_constant_folding=True,
        verbose=False
    )
    
    print(f"✅ ONNX exported: {onnx_path}")
    
    # Export thresholds
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    print(f"✅ Thresholds exported: {thresholds_path}")
    
    # Get file sizes
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"\n📊 Files created:")
    print(f"   - model.onnx: {onnx_size:.2f} MB")
    print(f"   - thresholds.json: {os.path.getsize(thresholds_path)} bytes")
    
    # Instructions for mobile
    print(f"\n📱 To use in your mobile app:")
    print(f"   Copy these files to:")
    print(f"   cp {onnx_path} ../twin-recognition-mobile/assets/model/model.onnx")
    print(f"   cp {thresholds_path} ../twin-recognition-mobile/assets/model/thresholds.json")

if __name__ == "__main__":
    # ============================================
    # LOAD YOUR TRAINED MODEL HERE
    # ============================================
    
    # Path to your trained model checkpoint
    CHECKPOINT_PATH = "checkpoints/checkpoint.pth"  # Adjust this!
    
    # Check if checkpoint exists
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ Checkpoint not found: {CHECKPOINT_PATH}")
        print("\nPlease update CHECKPOINT_PATH to your trained model file.")
        print("\nLooking for model files in checkpoints/...")
        
        # List available checkpoints
        if os.path.exists("checkpoints"):
            files = os.listdir("checkpoints")
            if files:
                print("\nAvailable checkpoint files:")
                for f in files:
                    if f.endswith(('.pth', '.pt', '.onnx')):
                        print(f"   - checkpoints/{f}")
            else:
                print("   No checkpoint files found in checkpoints/ directory")
        else:
            print("   checkpoints/ directory doesn't exist")
        exit(1)
    
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    # Initialize model architecture (adjust based on your actual model)
    # Make sure this matches your training architecture!
    model = SiameseNetwork()
    
    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded model weights")
    
    model.to(DEVICE)
    model.eval()
    
    # Define thresholds (adjust based on your validation results)
    thresholds = {
        'threshold_same': 0.35,  # Distance < this = same person
        'threshold_twin': 0.60   # Distance < this = twins
    }
    
    print(f"\n🎯 Using thresholds:")
    print(f"   Same person threshold: {thresholds['threshold_same']}")
    print(f"   Twins threshold: {thresholds['threshold_twin']}")
    
    # Export for mobile
    export_full_model(
        model, 
        thresholds,
        onnx_path="model.onnx",
        thresholds_path="thresholds.json"
    )