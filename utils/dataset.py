import os
import random
from PIL import Image
from torch.utils.data import Dataset


class PairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = os.listdir(root_dir)
        self.class_to_images = {}

        for cls in self.classes:
            path = os.path.join(root_dir, cls)
            self.class_to_images[cls] = os.listdir(path)

        self.all_classes = list(self.class_to_images.keys())

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        if random.random() > 0.5:
            # same
            cls = random.choice(self.all_classes)
            img1, img2 = random.sample(self.class_to_images[cls], 2)
            label = 1
        else:
            # different
            cls1, cls2 = random.sample(self.all_classes, 2)
            img1 = random.choice(self.class_to_images[cls1])
            img2 = random.choice(self.class_to_images[cls2])
            label = 0

        path1 = os.path.join(self.root_dir, cls if label else cls1, img1)
        path2 = os.path.join(self.root_dir, cls if label else cls2, img2)

        img1 = Image.open(path1).convert("RGB")
        img2 = Image.open(path2).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, float(label)