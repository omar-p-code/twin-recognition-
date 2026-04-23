import os
import torch

# ================== Base Directory ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== Paths ==================
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================== Hyperparameters ==================
IMG_SIZE = 128
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
MARGIN = 2.0
THRESHOLD = 0.5

# ================== Face Detection Params ==================
MIN_FACE_SIZE = 40
ADD_PADDING = 20
DETECTION_CONFIDENCE = 0.9

# ================== Device ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")