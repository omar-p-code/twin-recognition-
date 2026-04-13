import os
import torch

# ================== Base Directory ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ================== Paths ==================
DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# create checkpoints folder automatically
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================== Hyperparameters ==================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.0001
MARGIN = 2.0
THRESHOLD = 0.4

# ================== Device ==================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== Dataset Notes ==================
# data/train/person1/img1.jpg