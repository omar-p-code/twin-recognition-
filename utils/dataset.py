import os
import random
from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """
    Yields (img1, img2, label) pairs for contrastive / binary training.
    label=1 → same person, label=0 → different people.
    """

    def __init__(self, root_dir, transform=None, hard_negative_prob=0.3, hard_negative_pairs=None):
        self.root_dir = root_dir
        self.transform = transform
        self.hard_negative_prob = hard_negative_prob
        self.hard_negative_pairs = hard_negative_pairs

        self.classes = os.listdir(root_dir)
        self.class_to_images = {
            cls: os.listdir(os.path.join(root_dir, cls))
            for cls in self.classes
        }
        self.all_classes   = list(self.class_to_images.keys())
        self.valid_classes = [c for c in self.all_classes if len(self.class_to_images[c]) >= 2]
        
        # ── Auto-detect twin folders as hard negatives ──────────────────────
        # Expected format:
        #   twins_001_A
        #   twins_001_B
        #
        # Both folders represent different identities that are genetically
        # similar, so they are injected into hard_negative_pairs.

        auto_twin_pairs = []

        twin_groups = {}

        for cls in self.all_classes:
            parts = cls.split("_")

            # Expect at least: twins 001 A
            if len(parts) >= 3 and parts[0].lower() == "twins":
                twin_id = "_".join(parts[:-1])   # twins_001
                suffix  = parts[-1].upper()      # A / B

                if suffix in ("A", "B"):
                    twin_groups.setdefault(twin_id, {})[suffix] = cls

        # Build pair list only if both A and B exist
        for twin_id, group in twin_groups.items():
            if "A" in group and "B" in group:
                auto_twin_pairs.append((group["A"], group["B"]))

        # Merge user-provided hard negatives with auto-detected twins
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
            # print(f'same person:{cls}')
            label = 1
        else:
            # Prioritize hard negatives (twins)
            if self.hard_negative_pairs and random.random() < 0.7:
                cls1, cls2 = random.choice(self.hard_negative_pairs)
                img1_name = random.choice(self.class_to_images[cls1])
                img2_name = random.choice(self.class_to_images[cls2])
                print(f'hard negatives twins:{cls1} and {cls2}')
                label = 0
            # Otherwise use random negatives
            else:
                cls1, cls2 = random.sample(self.all_classes, 2)
                img1_name = random.choice(self.class_to_images[cls1])
                img2_name = random.choice(self.class_to_images[cls2])
                label = 0
            
            # cls1, cls2 = random.sample(self.all_classes, 2)
            # if random.random() < self.hard_negative_prob and self.hard_negative_pairs:
            #     cls1, cls2 = random.choice(self.hard_negative_pairs)
            #     while cls1 == cls2:
            #         cls2 = random.choice(self.all_classes)
            # img1_name = random.choice(self.class_to_images[cls1])
            # img2_name = random.choice(self.class_to_images[cls2])
            # label = 0

        img1 = Image.open(os.path.join(self.root_dir, cls1, img1_name)).convert("RGB")
        img2 = Image.open(os.path.join(self.root_dir, cls2, img2_name)).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, float(label)


class TripletDataset(Dataset):
    """
    Yields (anchor, positive, negative) triplets for triplet-loss training.

    anchor   — random image from class A
    positive — different image from class A (same identity)
    negative — random image from class B ≠ A (different identity)

    This is the correct way to train the model to distinguish
    same / twins / different — the loss learns relative distances
    rather than relying on a fixed threshold.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir
        self.transform = transform

        self.class_to_images = {
            cls: [
                os.path.join(root_dir, cls, f)
                for f in os.listdir(os.path.join(root_dir, cls))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            for cls in os.listdir(root_dir)
        }

        # Triplet training requires at least 2 images per identity
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
        # Pick anchor class (must have ≥2 images)
        anchor_cls = random.choice(self.valid_classes)
        anchor_path, positive_path = random.sample(self.class_to_images[anchor_cls], 2)

        # Pick a different class for the negative
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
