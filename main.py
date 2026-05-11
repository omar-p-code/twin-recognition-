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


def main():
    parser = argparse.ArgumentParser(description="Twin Recognition App")
    parser.add_argument("--ui",    choices=["a", "b"], default="b",
                        help="UI design: a = Dark Scientific, b = Modern Minimal (default)")
    parser.add_argument("--train", action="store_true",
                        help="Retrain the model before launching the UI")
    args = parser.parse_args()

    if args.train:
        print("Starting training...")
        from training.train import train
        train()
        print("Training complete.")

    if args.ui == "a":
        print("Launching Design A — Dark Scientific")
        from ui.design_a import DesignAApp
        DesignAApp().run()
    else:
        print("Launching Design B — Modern Minimal")
        from ui.design_b import DesignBApp
        DesignBApp().run()


if __name__ == "__main__":
    main()
