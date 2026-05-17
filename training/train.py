import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork, TripletLoss, ContrastiveLoss
from utils.dataset import TripletDataset, PairDataset, auto_detect_twin_pairs
from config import (
    DATA_DIR, CHECKPOINT_DIR, IMG_SIZE, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, TRIPLET_MARGIN, NORMALIZE_MEAN, NORMALIZE_STD, DEVICE,
)

# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

# -----------------------------------------------------------------------
# Threshold calibration (using the random validation loader)
# -----------------------------------------------------------------------
def calibrate_thresholds(model, val_loader, device):
    try:
        from sklearn.metrics import roc_curve
        use_sklearn = True
    except ImportError:
        use_sklearn = False
        print("! sklearn not installed – using fallback thresholds")

    model.eval()
    same_dist, diff_dist = [], []

    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader, desc="Calibrating thresholds"):
            img1, img2 = img1.to(device), img2.to(device)
            e1, e2 = model(img1, img2)
            if e1.size(0) != e2.size(0):
                print(f"⚠️ Skipping batch – shape mismatch: {e1.size(0)} vs {e2.size(0)}")
                continue
            dists = torch.nn.functional.pairwise_distance(e1, e2).cpu().numpy()
            labels_np = label.cpu().numpy()
            for d, l in zip(dists, labels_np):
                (same_dist if l == 1 else diff_dist).append(float(d))

    same_dist = np.array(same_dist)
    diff_dist = np.array(diff_dist)

    if use_sklearn and len(same_dist) > 0 and len(diff_dist) > 0:
        all_dists = np.concatenate([same_dist, diff_dist])
        all_labels = np.concatenate([np.ones(len(same_dist)), np.zeros(len(diff_dist))])
        fpr, tpr, thresholds = roc_curve(all_labels, -all_dists)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        th_same = float(-thresholds[best_idx])
        th_twin = float((th_same + np.mean(diff_dist)) / 2)
    else:
        th_same = float(np.mean(same_dist) + np.std(same_dist) * 0.5) if len(same_dist) else 0.5
        th_twin = float((np.mean(same_dist) + np.mean(diff_dist)) / 2) if len(same_dist) and len(diff_dist) else 1.0

    print(f"\n📊 Threshold calibration:"
          f"\n  SAME  → mean={np.mean(same_dist):.4f}  std={np.std(same_dist):.4f}"
          f"\n  DIFF  → mean={np.mean(diff_dist):.4f}  std={np.std(diff_dist):.4f}"
          f"\n  th_same={th_same:.4f}  th_twin={th_twin:.4f}")
    model.train()
    return th_same, th_twin

# -----------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------
def load_checkpoint(model, optimizer, checkpoint_dir, device):
    path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.exists(path):
        print("No checkpoint found – starting from scratch")
        return 0, float("inf"), 0.35, 0.60

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("Warning: could not restore optimizer state")
    start_epoch = ckpt.get("epoch", -1) + 1
    best_loss = ckpt.get("best_loss", float("inf"))
    th_same = ckpt.get("threshold_same", 0.35)
    th_twin = ckpt.get("threshold_twin", 0.60)
    print(f"Resumed from epoch {start_epoch} | best_loss={best_loss:.4f} | same={th_same:.4f} twin={th_twin:.4f}")
    return start_epoch, best_loss, th_same, th_twin

def save_checkpoint(model, optimizer, epoch, loss, best_loss, th_same, th_twin, checkpoint_dir):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "threshold_same": th_same,
        "threshold_twin": th_twin,
    }
    torch.save(ckpt, os.path.join(checkpoint_dir, "checkpoint.pth"))
    torch.save(ckpt, os.path.join(checkpoint_dir, "checkpoint2.pth"))   # backup

def export_onnx(model, checkpoint_dir, device):
    try:
        model.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        torch.onnx.export(
            model, (dummy, dummy),
            os.path.join(checkpoint_dir, "model.onnx"),
            input_names=["img1", "img2"], output_names=["emb1", "emb2"],
            opset_version=17, dynamo=False,
        )
        print("✓ ONNX model exported")
        model.train()
    except Exception as e:
        print(f"ONNX export failed: {e}")

# -----------------------------------------------------------------------
# Validation (using PairDataset – random pairs)
# -----------------------------------------------------------------------
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            e1, e2 = model(img1, img2)
            total += criterion(e1, e2, label).item()
    model.train()
    return total / len(loader)

# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    # Auto-detect twin pairs (used in PairDataset for hard negatives)
    hard_negative_pairs = auto_detect_twin_pairs(train_dir)   # also use for validation if needed
    if hard_negative_pairs:
        print(f"✓ Detected {len(hard_negative_pairs)} twin pairs – will be used as hard negatives.")

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    # Training: TripletDataset (no change)
    train_dataset = TripletDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)   # reduced workers

    # Validation: PairDataset with random pairs (original behaviour)
    val_dataset = PairDataset(val_dir, transform=val_transform,
                              hard_negative_pairs=hard_negative_pairs)   # pass twin pairs for hard negatives
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)   # workers=0 to avoid warnings

    model = SiameseNetwork().to(DEVICE)
    criterion = TripletLoss(margin=TRIPLET_MARGIN)
    val_criterion = ContrastiveLoss(margin=2.0)   # for validation loss monitoring
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_epoch, best_loss, th_same, th_twin = load_checkpoint(
        model, optimizer, CHECKPOINT_DIR, DEVICE
    )

    print(f"\n🚀 Training on {DEVICE} | epochs={NUM_EPOCHS} | batch={BATCH_SIZE}")
    print(f"   Backbone: ResNet50 | embedding: 256-d | loss: TripletLoss(margin={TRIPLET_MARGIN})\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            emb_a = model.forward_once(anchor)
            emb_p = model.forward_once(positive)
            emb_n = model.forward_once(negative)
            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, val_criterion, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        # Recalibrate thresholds every 5 epochs
        if (epoch + 1) % 5 == 0:
            th_same, th_twin = calibrate_thresholds(model, val_loader, DEVICE)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, best_loss,
                            th_same, th_twin, CHECKPOINT_DIR)
            print(f"  ✓ New best model saved (loss={best_loss:.4f})")

    # Final calibration and export
    th_same, th_twin = calibrate_thresholds(model, val_loader, DEVICE)
    save_checkpoint(model, optimizer, NUM_EPOCHS - 1, best_loss, best_loss,
                    th_same, th_twin, CHECKPOINT_DIR)
    export_onnx(model, CHECKPOINT_DIR, DEVICE)
    print("\n✅ Training complete!")

if __name__ == "__main__":
    train()