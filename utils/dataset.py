import os
import random
from PIL import Image
from torch.utils.data import Dataset

class PairDataset(Dataset):
    def __init__(self, root_dir, transform=None, hard_negative_prob=0.3, hard_negative_pairs=None):
        self.root_dir = root_dir
        self.transform = transform
        self.hard_negative_prob = hard_negative_prob
        self.hard_negative_pairs = hard_negative_pairs

        self.classes = os.listdir(root_dir)
        self.class_to_images = {}

        for cls in self.classes:
            path = os.path.join(root_dir, cls)
            self.class_to_images[cls] = os.listdir(path)

        self.all_classes = list(self.class_to_images.keys())
        
        # Precompute valid classes (classes with at least 2 images)
        self.valid_classes = [c for c in self.all_classes if len(self.class_to_images[c]) >= 2]

    def __len__(self):
        return 20000  # Arbitrary large number for infinite sampling

    def __getitem__(self, idx):

        # SAME PERSON
        if random.random() < 0.5:
            # Pick a class that has at least 2 images
            cls = random.choice(self.valid_classes)
            img1, img2 = random.sample(self.class_to_images[cls], 2)
            class1 = cls
            class2 = cls
            label = 1

        else:
            # DIFFERENT PERSON
            # Start with random different classes
            cls1, cls2 = random.sample(self.all_classes, 2)
            
            # Hard negative attempt
            if random.random() < self.hard_negative_prob and self.hard_negative_pairs:
                # Use predefined hard negative pairs if available
                hard_pair = random.choice(self.hard_negative_pairs)
                cls1, cls2 = hard_pair[0], hard_pair[1]
            
            # Ensure they are different classes
            while cls1 == cls2:
                cls2 = random.choice(self.all_classes)
            
            img1 = random.choice(self.class_to_images[cls1])
            img2 = random.choice(self.class_to_images[cls2])
            class1 = cls1
            class2 = cls2
            label = 0

        # Build paths using the tracked class names
        path1 = os.path.join(self.root_dir, class1, img1)
        path2 = os.path.join(self.root_dir, class2, img2)

        # Load and convert images
        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, float(label)