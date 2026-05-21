from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load your trained SiameseNetwork ────────────────────────────────────────
from models.siamese import SiameseNetwork

model = SiameseNetwork()
model.load_state_dict(
    torch.load("../checkpoints/checkpoint.pth", map_location="cpu")["model_state_dict"]
)
model.eval()

# ── Image preprocessing (same as training) ──────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Cosine similarity thresholds (replace after calibration) ────────────────
TH_SAME = 0.75   # similarity > this → same person
TH_TWIN = 0.45   # similarity between this and TH_SAME → twins

# ── FastAPI app ─────────────────────────────────────────────────────────────
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# ── Allow all origins (for development) ────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/compare")
async def compare(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    contents1 = await img1.read()
    contents2 = await img2.read()
    image1 = Image.open(io.BytesIO(contents1)).convert("RGB")
    image2 = Image.open(io.BytesIO(contents2)).convert("RGB")

    t1 = transform(image1).unsqueeze(0)
    t2 = transform(image2).unsqueeze(0)

    with torch.no_grad():
        emb1, emb2 = model(t1, t2)
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)

        # Cosine similarity (dot product of normalized vectors)
        similarity = torch.sum(emb1 * emb2).item()
        similarity = max(-1.0, min(1.0, similarity))   # clamp to valid range

    # ── Classification based on cosine similarity ──────────────────────────
    if similarity >= TH_SAME:
        label = "same_person"
        confidence = 50 + (similarity - TH_SAME) / (1 - TH_SAME) * 49
    elif similarity >= TH_TWIN:
        label = "twins"
        confidence = 50 + (similarity - TH_TWIN) / (TH_SAME - TH_TWIN) * 32
    else:
        label = "different"
        confidence = 50 - (TH_TWIN - similarity) / (TH_TWIN + 1) * 30

    confidence = min(99, max(50, confidence))

    # Keep the key "distance" so the mobile app doesn't need changes
    return {
        "label": label,
        "confidence": confidence,
        "distance": similarity 
    }