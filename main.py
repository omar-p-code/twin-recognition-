"""
Twin Recognition — entry point.
Usage:
    python main.py            # defaults to Design B (Modern Minimal)
    python main.py --ui a     # Design A (Dark Scientific)
    python main.py --ui b     # Design B (Modern Minimal)
    python main.py --train    # retrain the model
"""

import argparse
import os
import sys

import torch


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    parser = argparse.ArgumentParser(description="Twin Recognition App")
    parser.add_argument("--ui",    choices=["a", "b"],
                        help="UI design: a = Dark Scientific, b = Modern Minimal (default)")
    parser.add_argument("--train", action="store_true",
                        help="Retrain the model before launching the UI")
    args = parser.parse_args()

    if args.train:
        print("Starting training...")
        from training.train import train
        train()
        print("Training complete.")
    
    if args.ui == "b":
        print("Launching Design A — Dark Scientific")
        from ui.design_a import DesignAApp
        DesignAApp().run()
    elif args.ui == 'b':
        print("Launching Design B — Modern Minimal")
        from ui.design_b import DesignBApp
        DesignBApp().run()
    else:
        print("Testing model on two sample images...")
        img1 = "data/val/Abdullah_Gul/Abdullah_Gul_0018.jpg"
        img2 = "data/val/Abdullah_Gul/Abdullah_Gul_0007.jpg"
        # Use TFLite if available, else PyTorch
        try:
            from utils.compare_tflite import compare_tflite, classify
            distance, _, _ = compare_tflite("output_folder/model_float16.tflite", img1, img2)
            label = classify(distance, 0.4, 0.7)
            print(f"TFLite → Distance: {distance:.4f}, Result: {label}")
        except Exception as e:
            print(f"TFLite failed: {e}")
            print("Falling back to PyTorch...")
            from utils.compare_tflite import compare_pytorch
            distance, _, _ = compare_pytorch("checkpoints/checkpoint.pth", img1, img2)
            label = classify(distance, .4, 0.7)
            print(f"PyTorch → Distance: {distance:.4f}, Result: {label}")


if __name__ == "__main__":
    main()
