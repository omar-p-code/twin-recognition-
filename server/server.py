from fastapi import FastAPI, File, UploadFile
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Load your model ─────────────────────────────────────────────────────────
from models.siamese import SiameseNetwork

model = SiameseNetwork()
model.load_state_dict(
    torch.load("../checkpoints/checkpoint.pth", map_location="cpu")["model_state_dict"]
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

TH_SAME = 0.44
TH_TWIN = 0.74

# ── The FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI()

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
        distance = torch.norm(emb1 - emb2).item()

    if distance < TH_SAME:
        label = "same_person"
        confidence = min(99, max(50, 95 - (distance / TH_SAME) * 20))
    elif distance < TH_TWIN:
        label = "twins"
        confidence = min(99, max(50, 82 - ((distance - TH_SAME) / (TH_TWIN - TH_SAME)) * 22))
    else:
        label = "different"
        confidence = min(99, max(50, 90 - (min(distance - TH_TWIN, 0.8) / 0.8) * 30))

    return {"label": label, "confidence": confidence, "distance": distance}