import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import numpy as np
import sys
from itertools import cycle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.siamese import SiameseNetwork, TripletLoss, ContrastiveLoss
from utils.dataset import (
    TripletDataset,
    PairDataset,
    TwinPairDataset,
    auto_detect_twin_pairs,
    get_strong_train_transform,      # <-- for twin pairs
)
from config import (
    DATA_DIR,
    CHECKPOINT_DIR,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    TRIPLET_MARGIN,
    CONTRASTIVE_MARGIN,               # <-- new margin for twins
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    DEVICE,
)

# -----------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------
def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
# Calibration (unchanged, already supports cosine)
# -----------------------------------------------------------------------
def calibrate_threshold(model, val_loader, device, percentile_low=5, percentile_high=95, metric='cosine'):
    model.eval()
    same_vals, twin_vals, diff_vals = [], [], []

    with torch.no_grad():
        for img1, img2, label in tqdm(val_loader, desc=f"Calibrating thresholds ({metric})"):
            img1, img2 = img1.to(device), img2.to(device)
            e1, e2 = model(img1, img2)
            e1 = torch.nn.functional.normalize(e1, p=2, dim=1)
            e2 = torch.nn.functional.normalize(e2, p=2, dim=1)

            if metric == 'euclidean':
                vals = torch.nn.functional.pairwise_distance(e1, e2).cpu().numpy()
            else:
                vals = torch.sum(e1 * e2, dim=1).cpu().numpy()

            labels_np = label.cpu().numpy()
            for v, l in zip(vals, labels_np):
                if l == 2:
                    same_vals.append(v)
                elif l == 1:
                    twin_vals.append(v)
                else:
                    diff_vals.append(v)

    same_vals = np.array(same_vals)
    twin_vals = np.array(twin_vals)
    diff_vals = np.array(diff_vals)

    def stats(arr):
        return f"mean={np.mean(arr):.4f} std={np.std(arr):.4f} p5={np.percentile(arr,5):.4f} p95={np.percentile(arr,95):.4f}"

    unit = "distance" if metric == 'euclidean' else "similarity"
    print(f"\n📊 {unit.capitalize()} distributions ({metric}):")
    print(f"  SAME:  {stats(same_vals)}")
    print(f"  TWIN:  {stats(twin_vals)}")
    print(f"  DIFF:  {stats(diff_vals)}")

    overlap_same_diff = np.min(same_vals) <= np.max(diff_vals) if metric == 'cosine' else np.max(same_vals) >= np.min(diff_vals)
    if overlap_same_diff:
        print(f"\n⚠️ WARNING: same and diff {unit} distributions overlap!")
    else:
        print("\n✅ Good separation: same and diff are separated.")

    if len(twin_vals) > 0:
        if metric == 'euclidean':
            th_same_twin = (np.percentile(same_vals, percentile_high) + np.percentile(twin_vals, percentile_low)) / 2
            th_twin_diff = (np.percentile(twin_vals, percentile_high) + np.percentile(diff_vals, percentile_low)) / 2
        else:
            th_same_twin = (np.percentile(same_vals, percentile_low) + np.percentile(twin_vals, percentile_high)) / 2
            th_twin_diff = (np.percentile(twin_vals, percentile_low) + np.percentile(diff_vals, percentile_high)) / 2
    else:
        if metric == 'euclidean':
            th_same_twin = np.percentile(same_vals, percentile_high) + 0.1
            th_twin_diff = (np.percentile(same_vals, percentile_high) + np.percentile(diff_vals, percentile_low)) / 2
        else:
            th_same_twin = np.percentile(same_vals, percentile_low) - 0.1
            th_twin_diff = (np.percentile(twin_vals, percentile_low) + np.percentile(diff_vals, percentile_high)) / 2

    # Ensure correct ordering
    if metric == 'euclidean':
        if th_same_twin >= th_twin_diff:
            print("⚠️ Warning: distance thresholds inverted. Adjusting.")
            mid = (th_same_twin + th_twin_diff) / 2
            th_same_twin = mid - 0.1
            th_twin_diff = mid + 0.1
    else:
        if th_same_twin <= th_twin_diff:
            print("⚠️ Warning: similarity thresholds inverted. Adjusting.")
            mid = (th_same_twin + th_twin_diff) / 2
            th_same_twin = mid + 0.1
            th_twin_diff = mid - 0.1

    print(f"\n📊 Final {unit} thresholds:")
    print(f"  th_same_twin = {th_same_twin:.4f}")
    print(f"  th_twin_diff = {th_twin_diff:.4f}")
    model.train()
    return th_same_twin, th_twin_diff, overlap_same_diff

# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------
def validate(model, loader, device):
    model.eval()
    total = 0.0
    criterion = ContrastiveLoss(margin=2.0)
    with torch.no_grad():
        for img1, img2, label in loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            e1, e2 = model(img1, img2)
            binary_label = (label == 2).float()
            total += criterion(e1, e2, binary_label).item()
    model.train()
    return total / len(loader)

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

def export_onnx(model, checkpoint_dir, device):
    try:
        model.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        torch.onnx.export(
            model, (dummy, dummy),
            os.path.join(checkpoint_dir, "model.onnx"),
            input_names=["img1", "img2"],
            output_names=["emb1", "emb2"],
            opset_version=17,
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
    print(f"✓ Detected {len(hard_negative_pairs)} twin pairs – will use as hard negatives.")

    train_transform = get_train_transform()
    val_transform = get_val_transform()
    strong_transform = get_strong_train_transform()   # for twin pairs

    # Triplet dataset – twins NOT used as negatives here (empty list)
    triplet_dataset = TripletDataset(
        train_dir,
        transform=train_transform,
        hard_twin_pairs=[]          # <-- twins removed from triplets
    )
    triplet_loader = DataLoader(triplet_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, pin_memory=True)

    # Twin pair dataset – STRONG augmentations
    twin_dataset = TwinPairDataset(train_dir, hard_negative_pairs,
                                   transform=strong_transform)
    twin_loader = DataLoader(twin_dataset, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=True) if len(twin_dataset) > 0 else None

    # Validation dataset
    val_dataset = PairDataset(val_dir, transform=val_transform,
                              hard_negative_pairs=hard_negative_pairs)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = SiameseNetwork().to(DEVICE)
    criterion_triplet = TripletLoss(margin=TRIPLET_MARGIN)
    criterion_contrastive = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)   # <-- new margin

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    start_epoch, best_loss, th_same_twin, th_twin_diff = load_checkpoint(
        model, optimizer, CHECKPOINT_DIR, DEVICE
    )

    print(f"\n🚀 Hybrid training on {DEVICE} | epochs={NUM_EPOCHS}")
    if twin_loader:
        print(f"   Triplet + Contrastive (twins) | margin={TRIPLET_MARGIN} / contrastive margin={CONTRASTIVE_MARGIN}")
    else:
        print(f"   Triplet only (no twin pairs found) | margin={TRIPLET_MARGIN}")

    twin_iter = cycle(twin_loader) if twin_loader else None

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        steps = len(triplet_loader)

        progress_bar = tqdm(triplet_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", total=steps)
        for batch_idx, (anchor, positive, negative, *_) in enumerate(progress_bar):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

            # Triplet loss
            emb_a = model.forward_once(anchor)
            emb_p = model.forward_once(positive)
            emb_n = model.forward_once(negative)
            loss_triplet = criterion_triplet(emb_a, emb_p, emb_n)

            # Contrastive loss on strongly augmented twin pairs
            if twin_iter:
                try:
                    img1, img2, label = next(twin_iter)
                    img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
                    emb1 = model.forward_once(img1)
                    emb2 = model.forward_once(img2)
                    loss_contrastive = criterion_contrastive(emb1, emb2, label)
                except StopIteration:
                    loss_contrastive = 0.0
            else:
                loss_contrastive = 0.0

            loss = loss_triplet + 5.0 * loss_contrastive   # <-- boosted weight

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

            progress_bar.set_postfix(loss=loss.item(), triplet=loss_triplet.item(),
                                     twin=loss_contrastive.item() if twin_iter else 0.0)

        avg_loss = epoch_loss / steps
        val_loss = validate(model, val_loader, DEVICE)
        scheduler.step()

        print(f"Epoch {epoch+1:03d} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % 2 == 0:
            th_same_twin, th_twin_diff, _ = calibrate_threshold(model, val_loader, DEVICE, metric='cosine')

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss, best_loss,
                            th_same_twin, th_twin_diff, CHECKPOINT_DIR)
            print(f"  ✓ New best model saved (loss={best_loss:.4f})")

    # Final calibration & export
    th_same_twin, th_twin_diff, _ = calibrate_threshold(model, val_loader, DEVICE, metric='cosine')
    save_checkpoint(model, optimizer, NUM_EPOCHS-1, best_loss, best_loss,
                    th_same_twin, th_twin_diff, CHECKPOINT_DIR)
    export_onnx(model, CHECKPOINT_DIR, DEVICE)
    print("\n✅ Training complete!")

if __name__ == "__main__":
    train()