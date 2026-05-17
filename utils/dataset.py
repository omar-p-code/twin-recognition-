import os
import random
from PIL import Image
from torch.utils.data import Dataset

class PairDataset(Dataset):
    """
    Random pair dataset for validation.
    label = 2 → same person
    label = 1 → twin pair (hard negative)
    label = 0 → different people
    """
    def __init__(self, root_dir, transform=None, hard_negative_pairs=None):
        self.root_dir = root_dir
        self.transform = transform
        self.hard_negative_pairs = hard_negative_pairs or []

        # Build class -> list of image paths
        self.class_to_images = {}
        for cls in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if images:
                self.class_to_images[cls] = images

        self.all_classes = list(self.class_to_images.keys())
        self.valid_classes = [c for c in self.all_classes if len(self.class_to_images[c]) >= 2]

        # Auto-detect twin pairs from folder names
        auto_twin_pairs = []
        twin_groups = {}
        for cls in self.all_classes:
            parts = cls.split('_')
            if len(parts) >= 3 and parts[0].lower() == 'twins':
                twin_id = '_'.join(parts[:-1])
                suffix = parts[-1].upper()
                if suffix in ('A', 'B'):
                    twin_groups.setdefault(twin_id, {})[suffix] = cls
        for twin_id, group in twin_groups.items():
            if 'A' in group and 'B' in group:
                auto_twin_pairs.append((group['A'], group['B']))

        # Merge and keep only pairs that exist in this dataset
        if self.hard_negative_pairs:
            self.hard_negative_pairs.extend(auto_twin_pairs)
        else:
            self.hard_negative_pairs = auto_twin_pairs
        self.hard_negative_pairs = [
            (c1, c2) for (c1, c2) in self.hard_negative_pairs
            if c1 in self.class_to_images and c2 in self.class_to_images
        ]
        print(f"[PairDataset] Loaded {len(self.hard_negative_pairs)} valid hard negative twin pairs")

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        while True:
            try:
                r = random.random()
                if r < 0.33:
                    # Same person (label = 2)
                    if not self.valid_classes:
                        cls = random.choice(self.all_classes)
                        img_path = random.choice(self.class_to_images[cls])
                        img = Image.open(img_path).convert("RGB")
                        if self.transform:
                            img1 = self.transform(img)
                            img2 = self.transform(img)
                        else:
                            img1, img2 = img, img
                        label = 2
                        return img1, img2, float(label)
                    cls = random.choice(self.valid_classes)
                    img1_path, img2_path = random.sample(self.class_to_images[cls], 2)
                    img1 = Image.open(img1_path).convert("RGB")
                    img2 = Image.open(img2_path).convert("RGB")
                    label = 2

                elif r < 0.66:
                    # Twin pair (label = 1)
                    if not self.hard_negative_pairs:
                        # fallback to random different classes
                        cls1, cls2 = random.sample(self.all_classes, 2)
                    else:
                        cls1, cls2 = random.choice(self.hard_negative_pairs)
                    img1_path = random.choice(self.class_to_images[cls1])
                    img2_path = random.choice(self.class_to_images[cls2])
                    img1 = Image.open(img1_path).convert("RGB")
                    img2 = Image.open(img2_path).convert("RGB")
                    label = 1

                else:
                    # Different people (label = 0)
                    cls1, cls2 = random.sample(self.all_classes, 2)
                    while cls1 == cls2:
                        cls2 = random.choice(self.all_classes)
                    # Avoid accidentally picking a twin pair
                    while (cls1, cls2) in self.hard_negative_pairs or (cls2, cls1) in self.hard_negative_pairs:
                        cls2 = random.choice(self.all_classes)
                    img1_path = random.choice(self.class_to_images[cls1])
                    img2_path = random.choice(self.class_to_images[cls2])
                    img1 = Image.open(img1_path).convert("RGB")
                    img2 = Image.open(img2_path).convert("RGB")
                    label = 0

                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)
                return img1, img2, float(label)

            except FileNotFoundError as e:
                print(f"⚠️ Skipping missing file: {e.filename}")
                continue
            except Exception as e:
                print(f"⚠️ Skipping problematic pair: {e}")
                continue


class TripletDataset(Dataset):
    """Triplet dataset for training – with retry on missing files."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_images = {}
        for cls in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path):
                continue
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                self.class_to_images[cls] = images

        self.valid_classes = [c for c in self.class_to_images if len(self.class_to_images[c]) >= 2]
        self.all_classes = list(self.class_to_images.keys())

        if len(self.valid_classes) < 2:
            raise ValueError(
                f"TripletDataset needs at least 2 identities with ≥2 images each. "
                f"Found {len(self.valid_classes)} valid identities in {root_dir}."
            )

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        while True:
            try:
                anchor_cls = random.choice(self.valid_classes)
                anchor_path, positive_path = random.sample(self.class_to_images[anchor_cls], 2)
                negative_cls = random.choice([c for c in self.all_classes if c != anchor_cls])
                negative_path = random.choice(self.class_to_images[negative_cls])

                anchor = Image.open(anchor_path).convert("RGB")
                positive = Image.open(positive_path).convert("RGB")
                negative = Image.open(negative_path).convert("RGB")

                if self.transform:
                    anchor = self.transform(anchor)
                    positive = self.transform(positive)
                    negative = self.transform(negative)

                return anchor, positive, negative

            except FileNotFoundError as e:
                print(f"⚠️ Skipping missing file: {e.filename}")
                continue
            except Exception as e:
                print(f"⚠️ Skipping problematic triplet: {e}")
                continue


def auto_detect_twin_pairs(root_dir):
    """Detect twin folder pairs (e.g., twins_001_A, twins_001_B)."""
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