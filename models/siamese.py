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
    Siamese network with ResNet50 backbone.
    Removed Dropout for clean TFLite export.
    """
    def __init__(self):
        super().__init__()

        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Feature extractor: remove final classification head
        self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])

        # Projection head: 2048 → 512 → EMBEDDING_DIM (no Dropout)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),   # removed for TFLite compatibility
            nn.Linear(512, EMBEDDING_DIM),
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)   # flatten
        x = self.fc(x)
        # Optional L2 normalisation – uncomment if used during training
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2, reduction="mean")

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)


class ContrastiveLoss(nn.Module):
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
# ----- optional face comparison function (unchanged) -----
from utils.preprocess import preprocess_image

def compare_faces(model, img1, img2, th_same, th_twin, device="cpu", detect_faces=True):
    model.eval()
    try:
        from utils.face_detection import face_detector
    except ImportError:
        class _SimpleDetector:
            def detect_and_crop_face(self, img_path, padding=20):
                try:
                    img = Image.open(img_path).convert("RGB") if isinstance(img_path, str) else img_path
                    img.thumbnail((300,300))
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


class BatchHardTripletLoss(nn.Module):
    """
    Batch‑hard triplet loss: for each anchor, the hardest positive (farthest
    positive) and hardest negative (nearest negative) within the batch are used.
    Margin is enforced between the hardest positive distance and the hardest
    negative distance.
    """
    def __init__(self, margin=1.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        embeddings : [N, dim]  (L2‑normalised)
        labels : [N]           (integer class indices)
        """
        pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        mask_anchor_positive = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_anchor_negative = labels.unsqueeze(0) != labels.unsqueeze(1)

        # Hardest positive = largest distance among same‑class pairs
        hardest_positive_dist = pairwise_dist * mask_anchor_positive.float()
        hardest_positive_dist = hardest_positive_dist.max(dim=1)[0]

        # Hardest negative = smallest distance among different‑class pairs
        hardest_negative_dist = pairwise_dist + 1e9 * (~mask_anchor_negative).float()
        hardest_negative_dist = hardest_negative_dist.min(dim=1)[0]

        loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)
        return loss.mean()