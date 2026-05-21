import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from torch.utils.data import DataLoader
from utils.dataset import PairDataset, auto_detect_twin_pairs
from config import (
    DATA_DIR, IMG_SIZE, BATCH_SIZE, DEVICE, NORMALIZE_MEAN, NORMALIZE_STD
)

def main():
    # Load the model
    from models.siamese import SiameseNetwork
    model = SiameseNetwork().to(DEVICE)
    model.load_state_dict(torch.load("checkpoints/checkpoint.pth", map_location="cpu")["model_state_dict"])
    model.eval()

    # Prepare validation loader
    import torchvision.transforms as transforms
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])
    hard_negative_pairs = auto_detect_twin_pairs(DATA_DIR + "/train")
    val_dataset = PairDataset(DATA_DIR + "/val", transform=val_transform,
                              hard_negative_pairs=hard_negative_pairs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Collect cosine similarities
    same_sims, twin_sims, diff_sims = [], [], []
    with torch.no_grad():
        for img1, img2, label in val_loader:
            e1, e2 = model(img1.to(DEVICE), img2.to(DEVICE))
            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
            sims = torch.sum(e1 * e2, dim=1).cpu().numpy()
            for s, l in zip(sims, label.numpy()):
                if l == 2:   same_sims.append(s)
                elif l == 1: twin_sims.append(s)
                else:        diff_sims.append(s)

    # Compute thresholds using percentiles
    th_same = (np.percentile(same_sims, 5) + np.percentile(twin_sims, 95)) / 2
    th_twin = (np.percentile(twin_sims, 5) + np.percentile(diff_sims, 95)) / 2

    print(f"Cosine thresholds → TH_SAME: {th_same:.4f}, TH_TWIN: {th_twin:.4f}")

if __name__ == "__main__":
    main()