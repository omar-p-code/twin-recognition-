import torch

# ================== Base Directory ==================
BASE_DIR = r"Y:\dev\myProjects\twin-recognition"

# ================== Paths ==================
DATA_DIR = f"{BASE_DIR}/data"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"

# ================== Hyperparameters ==================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.0001
MARGIN = 2.0
THRESHOLD = 0.4  # Distance below this → same person (Twin)

# ================== Device ==================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================== Dataset Notes ==================
# Dataset structure example:
# data/train/person1/img1.jpg
# data/train/person2/img2.jpg