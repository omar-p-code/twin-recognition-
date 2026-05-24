import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from config import IMG_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

# ─── Strong augmentations for twin pairs (to prevent overfitting) ──────────
def get_strong_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

# ──────────────────────────────────────────────────────────────────────
# 1. PairDataset (unchanged, for validation)
# ──────────────────────────────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, root_dir, transform=None, hard_negative_pairs=None):
        self.root_dir = root_dir
        self.transform = transform
        self.hard_negative_pairs = hard_negative_pairs or []

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
                    if not self.hard_negative_pairs:
                        cls1, cls2 = random.sample(self.all_classes, 2)
                    else:
                        cls1, cls2 = random.choice(self.hard_negative_pairs)
                    img1_path = random.choice(self.class_to_images[cls1])
                    img2_path = random.choice(self.class_to_images[cls2])
                    img1 = Image.open(img1_path).convert("RGB")
                    img2 = Image.open(img2_path).convert("RGB")
                    label = 1

                else:
                    cls1, cls2 = random.sample(self.all_classes, 2)
                    while cls1 == cls2:
                        cls2 = random.choice(self.all_classes)
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

            except (FileNotFoundError, Exception) as e:
                print(f"⚠️ Skipping pair due to: {e}")
                continue


# ──────────────────────────────────────────────────────────────────────
# 2. TripletDataset (returns labels for batch‑hard)
# ──────────────────────────────────────────────────────────────────────
class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None, hard_twin_pairs=None):
        self.root_dir = root_dir
        self.transform = transform

        self.class_to_images = {}
        for cls in os.listdir(root_dir):
            cls_path = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_path): continue
            images = [os.path.join(cls_path, f) for f in os.listdir(cls_path)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if images:
                self.class_to_images[cls] = images

        self.anchor_classes = [c for c in self.class_to_images if len(self.class_to_images[c]) >= 2]
        if not self.anchor_classes:
            raise ValueError("Need at least one class with ≥2 images for anchor.")

        self.all_classes = list(self.class_to_images.keys())

        # Build a mapping class -> index (for label)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.all_classes)}

        self.hard_twin_pairs = hard_twin_pairs or []
        self.hard_twin_pairs = [(a,b) for (a,b) in self.hard_twin_pairs
                                if a in self.class_to_images and b in self.class_to_images]

        print(f"[TripletDataset] {len(self.anchor_classes)} anchor classes, "
              f"{len(self.hard_twin_pairs)} twin pairs as negatives")

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        # 1. Pick anchor class (always ≥2 images)
        anchor_cls = random.choice(self.anchor_classes)

        # 2. Decide whether to use a twin hard negative (30% chance)
        use_twin = self.hard_twin_pairs and random.random() < 0.3
        if use_twin:
            a_twin, b_twin = random.choice(self.hard_twin_pairs)
            possible = [c for c in (a_twin, b_twin) if c != anchor_cls and c in self.class_to_images]
            if possible:
                negative_cls = random.choice(possible)
            else:
                use_twin = False
        if not use_twin:
            negative_cls = random.choice([c for c in self.all_classes if c != anchor_cls])

        # 3. Sample images
        anchor_path, positive_path = random.sample(self.class_to_images[anchor_cls], 2)
        negative_path = random.choice(self.class_to_images[negative_cls])

        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        # Return class indices for anchor and negative
        anchor_label = self.class_to_idx[anchor_cls]
        negative_label = self.class_to_idx[negative_cls]

        return anchor, positive, negative, anchor_label, negative_label


# ──────────────────────────────────────────────────────────────────────
# 3. TwinPairDataset (uses strong augmentations)
# ──────────────────────────────────────────────────────────────────────
class TwinPairDataset(Dataset):
    def __init__(self, root_dir, twin_pairs, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.twin_pairs = []

        for class_a, class_b in twin_pairs:
            dir_a = os.path.join(root_dir, class_a)
            dir_b = os.path.join(root_dir, class_b)

            if not os.path.isdir(dir_a) or not os.path.isdir(dir_b):
                continue

            imgs_a = [
                os.path.join(dir_a, f)
                for f in os.listdir(dir_a)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            imgs_b = [
                os.path.join(dir_b, f)
                for f in os.listdir(dir_b)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            if len(imgs_a) and len(imgs_b):
                self.twin_pairs.append((imgs_a, imgs_b))

        print(
            f"[TwinPairDataset] Loaded {len(self.twin_pairs)} twin groups"
        )

    def __len__(self):
        return len(self.twin_pairs)

    def __getitem__(self, idx):
        imgs_a, imgs_b = self.twin_pairs[idx]

        img_a_path = random.choice(imgs_a)
        img_b_path = random.choice(imgs_b)

        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")

        if self.transform:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)

        label = torch.tensor(1.0, dtype=torch.float32)

        return img_a, img_b, label
# ──────────────────────────────────────────────────────────────────────
# 4. auto_detect_twin_pairs (unchanged)
# ──────────────────────────────────────────────────────────────────────
def auto_detect_twin_pairs(root_dir):
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