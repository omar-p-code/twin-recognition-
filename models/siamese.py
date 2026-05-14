import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import EMBEDDING_DIM, IMG_SIZE


class SiameseNetwork(nn.Module):
    """
    Siamese network with a ResNet50 backbone.
    ResNet50 produces 2048-dim features vs ResNet18's 512,
    giving the model much more capacity for fine-grained twin discrimination.
    """

    def __init__(self):
        super().__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Strip the final classification head — keep the feature extractor
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # Projection head: 2048 → 512 → EMBEDDING_DIM (256)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, EMBEDDING_DIM),
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   # flatten: (B, 2048)
        x = self.fc(x)
        # x = F.normalize(x, p=2, dim=1)  # L2-normalise → unit sphere
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


# ================== TRIPLET LOSS ==================
class TripletLoss(nn.Module):
    """
    Triplet Loss: pulls anchor closer to positive, pushes it away from negative.

    Loss = max(d(a,p) - d(a,n) + margin, 0)

    This is far better than contrastive loss for learning structured
    embedding spaces — the model learns relative ordering rather than
    a fixed threshold.
    """

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


# ================== CONTRASTIVE LOSS (kept for compatibility) ==================
class ContrastiveLoss(nn.Module):
    """
    Legacy contrastive loss — kept so existing checkpoints can still be loaded.
    New training should use TripletLoss.
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = torch.mean(
            label * torch.pow(dist, 2)
            + (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )
        return loss


# ================== DATA PREPROCESSING ==================
from utils.preprocess import preprocess_image


# ================== FACE COMPARISON ==================
def compare_faces(model, img1, img2, th_same, th_twin, device="cpu", detect_faces=True):
    """Compare two face images and return (distance, label, detection_info)."""
    model.eval()

    try:
        from utils.face_detection import face_detector
    except ImportError:
        class _SimpleDetector:
            def detect_and_crop_face(self, img_path, padding=20):
                try:
                    img = Image.open(img_path).convert("RGB") if isinstance(img_path, str) else img_path
                    img.thumbnail((300, 300))
                    return img, "Fallback detection"
                except Exception:
                    return None, "Error"
        face_detector = _SimpleDetector()

    detection_info = {}

    if detect_faces:
        face1, msg1 = face_detector.detect_and_crop_face(img1)
        face2, msg2 = face_detector.detect_and_crop_face(img2)

        detection_info["face1_detected"] = face1 is not None
        detection_info["face2_detected"] = face2 is not None
        detection_info["face1_message"]  = msg1
        detection_info["face2_message"]  = msg2

        if face1 is None or face2 is None:
            return None, None, detection_info

        img1_proc, img2_proc = face1, face2
    else:
        img1_proc = Image.open(img1).convert("RGB") if isinstance(img1, str) else img1
        img2_proc = Image.open(img2).convert("RGB") if isinstance(img2, str) else img2

    try:
        t1 = preprocess_image(img1_proc).to(device)
        t2 = preprocess_image(img2_proc).to(device)

        with torch.no_grad():
            e1 = model.forward_once(t1)
            e2 = model.forward_once(t2)
            distance = F.pairwise_distance(e1, e2).item()

        if distance < th_same:
            result = "Same Person"
        elif distance < th_twin:
            result = "Twins"
        else:
            result = "Different People"

        return distance, result, detection_info

    except Exception as e:
        print(f"Comparison error: {e}")
        return None, "Comparison Failed", detection_info
