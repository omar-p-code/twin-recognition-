import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import sys
from sklearn.metrics import roc_curve

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
# 3‑label threshold calibration
# -----------------------------------------------------------------------
def calibrate_threshold(model, val_loader, device):
    model.eval()
    same_dists, twin_dists, diff_dists = [], [], []

    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader, desc="Calibrating thresholds"):
            img1, img2 = img1.to(device), img2.to(device)
            e1, e2 = model(img1, img2)
            # Force L2 normalisation if your model doesn't already do it
            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
            dists = torch.nn.functional.pairwise_distance(e1, e2).cpu().numpy()
            labels_np = label.cpu().numpy()
            for d, l in zip(dists, labels_np):
                if l == 2:
                    same_dists.append(d)
                elif l == 1:
                    twin_dists.append(d)
                else:
                    diff_dists.append(d)

    same_dists = np.array(same_dists)
    twin_dists = np.array(twin_dists)
    diff_dists = np.array(diff_dists)

    print(f"Same:  mean={np.mean(same_dists):.4f} max={np.max(same_dists):.4f}")
    print(f"Twin:  mean={np.mean(twin_dists):.4f} min={np.min(twin_dists):.4f} max={np.max(twin_dists):.4f}")
    print(f"Diff:  mean={np.mean(diff_dists):.4f} min={np.min(diff_dists):.4f}")

    # Sanity check (fail if no separation)
    if np.max(same_dists) >= np.min(diff_dists):
        print("⚠️ WARNING: Model has not learned proper separation (same/diff overlap).")
        # Still compute thresholds but they will be unreliable

    # Stable max‑min thresholds
    th_same_twin = (np.max(same_dists) + np.min(twin_dists)) / 2 if len(twin_dists) else np.max(same_dists) + 0.1
    th_twin_diff = (np.max(twin_dists) + np.min(diff_dists)) / 2 if len(twin_dists) else (np.max(same_dists) + np.min(diff_dists)) / 2

    # Optional: clip to reasonable range [0, 2] for normalised embeddings
    th_same_twin = np.clip(th_same_twin, 0.1, 1.5)
    th_twin_diff = np.clip(th_twin_diff, 0.2, 1.8)

    print(f"\n📊 Final thresholds:")
    print(f"  th_same_twin = {th_same_twin:.4f}  (distance < this → SAME PERSON)")
    print(f"  th_twin_diff = {th_twin_diff:.4f}   (distance between → TWINS, > this → DIFFERENT)")

    model.train()
    return th_same_twin, th_twin_diff
# -----------------------------------------------------------------------
# Validation (binary for monitoring)
# -----------------------------------------------------------------------
def validate(model, loader, device):
    """Monitors contrastive loss: treat same (2) as positive, twins(1)/diff(0) as negative."""
    model.eval()
    total = 0.0
    criterion = ContrastiveLoss(margin=2.0)
    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            e1, e2 = model(img1, img2)
            # Map label: 2 → 1 (positive), 0 or 1 → 0 (negative)
            binary_label = (label == 2).float()
            total += criterion(e1, e2, binary_label).item()
    model.train()
    return total / len(loader)

# -----------------------------------------------------------------------
# Checkpoint helpers (store two thresholds)
# -----------------------------------------------------------------------
def load_checkpoint(model, optimizer, checkpoint_dir, device):
    path = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.exists(path):
        print("No checkpoint found – starting from scratch")
        return 0, float("inf"), 0.35, 0.60   # fallback thresholds

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            print("Warning: could not restore optimizer state")
    start_epoch = ckpt.get("epoch", -1) + 1
    best_loss = ckpt.get("best_loss", float("inf"))
    th_same_twin = ckpt.get("threshold_same_twin", 0.35)
    th_twin_diff = ckpt.get("threshold_twin_diff", 0.60)
    print(f"Resumed from epoch {start_epoch} | best_loss={best_loss:.4f}")
    print(f"  thresholds: same_twin={th_same_twin:.4f}, twin_diff={th_twin_diff:.4f}")
    return start_epoch, best_loss, th_same_twin, th_twin_diff

def save_checkpoint(model, optimizer, epoch, loss, best_loss, th_same_twin, th_twin_diff, checkpoint_dir):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_loss": best_loss,
        "threshold_same_twin": th_same_twin,
        "threshold_twin_diff": th_twin_diff,
    }
    torch.save(ckpt, os.path.join(checkpoint_dir, "checkpoint.pth"))
    torch.save(ckpt, os.path.join(checkpoint_dir, "checkpoint2.pth"))

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
# Main training loop
# -----------------------------------------------------------------------
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")

    hard_negative_pairs = auto_detect_twin_pairs(train_dir)
    if hard_negative_pairs:
        print(f"✓ Detected {len(hard_negative_pairs)} twin pairs – will be used as hard negatives.")

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_dataset = TripletDataset(train_dir, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)

    val_dataset = PairDataset(val_dir, transform=val_transform,
                              hard_negative_pairs=hard_negative_pairs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = SiameseNetwork().to(DEVICE)
    criterion = TripletLoss(margin=TRIPLET_MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_epoch, best_loss, th_same_twin, th_twin_diff = load_checkpoint(
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
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 5 == 0:
            th_same_twin, th_twin_diff = calibrate_threshold(model, val_loader, DEVICE)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, best_loss,
                            th_same_twin, th_twin_diff, CHECKPOINT_DIR)
            print(f"  ✓ New best model saved (loss={best_loss:.4f})")

    # Final calibration and export
    th_same_twin, th_twin_diff = calibrate_threshold(model, val_loader, DEVICE)
    save_checkpoint(model, optimizer, NUM_EPOCHS - 1, best_loss, best_loss,
                    th_same_twin, th_twin_diff, CHECKPOINT_DIR)
    export_onnx(model, CHECKPOINT_DIR, DEVICE)
    print("\n✅ Training complete!")

if __name__ == "__main__":
    train()