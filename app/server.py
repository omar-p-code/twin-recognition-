from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from pathlib import Path
import io
import sys

# ---------------------------------------------------
# Project root
# ---------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# ---------------------------------------------------
# Import model
# ---------------------------------------------------
from models.siamese import SiameseNetwork

# ---------------------------------------------------
# Checkpoint path
# ---------------------------------------------------
CHECKPOINT_PATH = ROOT_DIR / "checkpoints" / "checkpoint_clean.pth"

print(f"\nLoading checkpoint from: {CHECKPOINT_PATH}")

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(
        f"Checkpoint not found at {CHECKPOINT_PATH}"
    )

# ---------------------------------------------------
# Load checkpoint
# ---------------------------------------------------
# import numpy
# torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=torch.device("cpu"),
    weights_only=False
)

# ---------------------------------------------------
# Load model
# ---------------------------------------------------
model = SiameseNetwork()

state_dict = checkpoint.get(
    "model_state_dict",
    checkpoint
)

model.load_state_dict(state_dict)
model.eval()

print("Model loaded successfully.")

# ---------------------------------------------------
# Load thresholds
# ---------------------------------------------------
TH_SAME = checkpoint.get("threshold_same_twin", 0.75)
TH_TWIN = checkpoint.get("threshold_twin_diff", 0.55)

print(f"TH_SAME = {TH_SAME}")
print(f"TH_TWIN = {TH_TWIN}")

# ---------------------------------------------------
# Image preprocessing
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# ---------------------------------------------------
# FastAPI app
# ---------------------------------------------------
app = FastAPI(
    title="Twin Recognition API"
)

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Root endpoint
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Twin Recognition API running"
    }

# ---------------------------------------------------
# Health endpoint
# ---------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "checkpoint_loaded": True,
        "threshold_same": TH_SAME,
        "threshold_twin": TH_TWIN,
    }

# ---------------------------------------------------
# Compare endpoint
# ---------------------------------------------------
@app.post("/compare")
async def compare(
    img1: UploadFile = File(...),
    img2: UploadFile = File(...),
):
    try:
        contents1 = await img1.read()
        contents2 = await img2.read()

        image1 = Image.open(
            io.BytesIO(contents1)
        ).convert("RGB")

        image2 = Image.open(
            io.BytesIO(contents2)
        ).convert("RGB")

        t1 = transform(image1).unsqueeze(0)
        t2 = transform(image2).unsqueeze(0)

        with torch.no_grad():
            emb1, emb2 = model(t1, t2)

            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            similarity = F.cosine_similarity(
                emb1,
                emb2
            ).item()

        similarity = max(
            -1.0,
            min(1.0, similarity)
        )

        if similarity >= TH_SAME:
            label = "same_person"
        elif similarity >= TH_TWIN:
            label = "twins"
        else:
            label = "different"

        confidence = round(similarity * 100, 2)

        print("\n--- Compare Request ---")
        print(f"Similarity: {similarity:.4f}")
        print(f"Prediction: {label}")
        print("-----------------------\n")

        return {
            "label": label,
            "confidence": confidence,
            "distance": round(similarity, 4),
        }

    except Exception as e:
        print(f"Compare endpoint error: {e}")
        return {
            "error": str(e)
        }