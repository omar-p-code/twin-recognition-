import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# Image
# =========================
IMG_SIZE = 224  # upgraded from 128 — preserves more facial detail

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# =========================
# Training
# =========================
BATCH_SIZE     = 32
NUM_EPOCHS     = 40
EMBEDDING_DIM  = 256      # upgraded from 64 — richer face representation
LEARNING_RATE  = 1e-4
MARGIN         = 2.0      # contrastive loss margin (kept for compatibility)
TRIPLET_MARGIN = 0.5      # margin for triplet loss
SAME_LABEL     = 1
DIFF_LABEL     = 0

# =========================
# Face Detection
# =========================
MIN_FACE_SIZE          = 40
ADD_PADDING            = 20
DETECTION_CONFIDENCE   = 0.9

# =========================
# Device
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
