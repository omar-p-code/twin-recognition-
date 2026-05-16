import os
import random
from PIL import Image
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """
    Yields (img1, img2, label) pairs for contrastive / binary training.
    label=1 → same person, label=0 → different people.
    (Not used in triplet training, but kept for compatibility.)
    """
    def __init__(self, root_dir, transform=None, hard_negative_prob=0.3, hard_negative_pairs=None):
        self.root_dir = root_dir
        self.transform = transform
        self.hard_negative_prob = hard_negative_prob
        self.hard_negative_pairs = hard_negative_pairs

        self.classes = os.listdir(root_dir)
        self.class_to_images = {
            cls: [f for f in os.listdir(os.path.join(root_dir, cls)) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            for cls in self.classes
        }
        self.all_classes   = list(self.class_to_images.keys())
        self.valid_classes = [c for c in self.all_classes if len(self.class_to_images[c]) >= 2]
        
        # Auto-detect twin folders as hard negatives
        auto_twin_pairs = []
        twin_groups = {}
        for cls in self.all_classes:
            parts = cls.split("_")
            if len(parts) >= 3 and parts[0].lower() == "twins":
                twin_id = "_".join(parts[:-1])
                suffix  = parts[-1].upper()
                if suffix in ("A", "B"):
                    twin_groups.setdefault(twin_id, {})[suffix] = cls
        for twin_id, group in twin_groups.items():
            if "A" in group and "B" in group:
                auto_twin_pairs.append((group["A"], group["B"]))

        if self.hard_negative_pairs:
            self.hard_negative_pairs.extend(auto_twin_pairs)
        else:
            self.hard_negative_pairs = auto_twin_pairs

        print(f"[PairDataset] Loaded {len(self.hard_negative_pairs)} hard negative twin pairs")

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        if random.random() < 0.5:
            cls = random.choice(self.valid_classes)
            img1_name, img2_name = random.sample(self.class_to_images[cls], 2)
            cls1 = cls2 = cls
            label = 1
        else:
            if self.hard_negative_pairs and random.random() < 0.7:
                cls1, cls2 = random.choice(self.hard_negative_pairs)
                if len(self.class_to_images[cls1]) == 0 or len(self.class_to_images[cls2]) == 0:
                    cls1, cls2 = random.sample(self.all_classes, 2)
                img1_name = random.choice(self.class_to_images[cls1])
                img2_name = random.choice(self.class_to_images[cls2])
            else:
                cls1, cls2 = random.sample(self.all_classes, 2)
                while len(self.class_to_images[cls1]) == 0:
                    cls1 = random.choice(self.all_classes)
                while len(self.class_to_images[cls2]) == 0:
                    cls2 = random.choice(self.all_classes)
                img1_name = random.choice(self.class_to_images[cls1])
                img2_name = random.choice(self.class_to_images[cls2])
            label = 0

        img1 = Image.open(os.path.join(self.root_dir, cls1, img1_name)).convert("RGB")
        img2 = Image.open(os.path.join(self.root_dir, cls2, img2_name)).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, float(label)


class TripletDataset(Dataset):
    """Yields (anchor, positive, negative) triplets for triplet-loss training."""
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform

        self.class_to_images = {
            cls: [os.path.join(root_dir, cls, f) for f in os.listdir(os.path.join(root_dir, cls))
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for cls in os.listdir(root_dir)
        }
        self.valid_classes = [c for c, imgs in self.class_to_images.items() if len(imgs) >= 2]
        self.all_classes   = list(self.class_to_images.keys())

        if len(self.valid_classes) < 2:
            raise ValueError(
                f"TripletDataset needs at least 2 identities with ≥2 images each. "
                f"Found {len(self.valid_classes)} valid identities in {root_dir}."
            )

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        anchor_cls = random.choice(self.valid_classes)
        anchor_path, positive_path = random.sample(self.class_to_images[anchor_cls], 2)

        negative_cls = random.choice([c for c in self.all_classes if c != anchor_cls])
        negative_path = random.choice(self.class_to_images[negative_cls])

        anchor   = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor   = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class FixedPairDataset(Dataset):
    """Deterministic dataset for validation: pre‑generated list of (path1, path2, label)."""
    def __init__(self, pairs, transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, float(label)


def auto_detect_twin_pairs(root_dir):
    """
    Scan folder names in root_dir and detect twin pairs based on pattern:
    e.g., "twins_001_A" and "twins_001_B" → pair ("twins_001_A", "twins_001_B")
    Returns a list of tuples (classA, classB).
    """
    classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    twin_groups = {}
    for cls in classes:
        parts = cls.split('_')
        if len(parts) >= 3 and parts[0].lower() == 'twins':
            group_id = '_'.join(parts[:-1])
            suffix = parts[-1].upper()
            if suffix in ('A', 'B'):
                twin_groups.setdefault(group_id, {})[suffix] = cls
    pairs = []
    for group_id, group in twin_groups.items():
        if 'A' in group and 'B' in group:
            pairs.append((group['A'], group['B']))
    print(f"[auto_detect_twin_pairs] Found {len(pairs)} twin pairs in {root_dir}")
    return pairs


def build_fixed_val_pairs(val_dir, num_pairs=2000, hard_negative_pairs=None):
    """
    Generate a fixed list of (img1_path, img2_path, label) for validation.
    Uses the same logic as PairDataset but without randomness per epoch.
    """
    # Build class -> images mapping
    class_to_images = {}
    for cls in os.listdir(val_dir):
        cls_path = os.path.join(val_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                  if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if images:
            class_to_images[cls] = images

    all_classes = list(class_to_images.keys())
    valid_classes = [c for c in all_classes if len(class_to_images[c]) >= 2]

    # Auto-detect twin pairs if not provided
    if hard_negative_pairs is None:
        hard_negative_pairs = auto_detect_twin_pairs(val_dir)

    pairs = []
    for _ in range(num_pairs):
        if random.random() < 0.5:
            # Same person
            if valid_classes:
                cls = random.choice(valid_classes)
                img1_path = random.choice(class_to_images[cls])
                img2_path = random.choice([p for p in class_to_images[cls] if p != img1_path])
            else:
                # Fallback: use same image twice
                cls = random.choice(all_classes)
                img1_path = random.choice(class_to_images[cls])
                img2_path = img1_path
            label = 1
        else:
            # Different person
            if hard_negative_pairs and random.random() < 0.7:
                cls1, cls2 = random.choice(hard_negative_pairs)
                if cls1 not in class_to_images or cls2 not in class_to_images:
                    cls1, cls2 = random.sample(all_classes, 2)
            else:
                cls1, cls2 = random.sample(all_classes, 2)
            # Ensure different classes
            while cls1 == cls2:
                cls2 = random.choice(all_classes)
            img1_path = random.choice(class_to_images[cls1])
            img2_path = random.choice(class_to_images[cls2])
            label = 0
        pairs.append((img1_path, img2_path, label))
    return pairs