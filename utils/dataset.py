import os
import random
from torch.utils.data import Dataset
from PIL import Image


class TripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.normal_people = {}
        self.twin_pairs = []

        # read normal people
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)

            if folder == "twins":
                continue

            if os.path.isdir(folder_path):
                images = [
                    os.path.join(folder_path, img)
                    for img in os.listdir(folder_path)
                    if img.lower().endswith(
                        (".jpg", ".jpeg", ".png")
                    )
                ]

                if len(images) >= 2:
                    self.normal_people[folder] = images

        # read twin pairs
        twins_dir = os.path.join(root_dir, "twins")

        if os.path.exists(twins_dir):
            for pair_folder in os.listdir(twins_dir):
                pair_path = os.path.join(
                    twins_dir,
                    pair_folder
                )

                if os.path.isdir(pair_path):
                    twin_folders = [
                        os.path.join(pair_path, twin)
                        for twin in os.listdir(pair_path)
                        if os.path.isdir(
                            os.path.join(pair_path, twin)
                        )
                    ]

                    if len(twin_folders) == 2:
                        self.twin_pairs.append(
                            twin_folders
                        )

        self.people_keys = list(
            self.normal_people.keys()
        )

        print("Normal people:", len(self.people_keys))
        print("Twin pairs:", len(self.twin_pairs))

    def __len__(self):
        return max(
            len(self.people_keys) * 20,
            len(self.twin_pairs) * 20
        )

    def load_image(self, path):
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, index):
        use_twins = random.random() < 0.4 and len(self.twin_pairs) > 0

        # ===== twin case =====
        if use_twins:
            pair = random.choice(self.twin_pairs)

            twin1_imgs = [
                os.path.join(pair[0], img)
                for img in os.listdir(pair[0])
                if img.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            twin2_imgs = [
                os.path.join(pair[1], img)
                for img in os.listdir(pair[1])
                if img.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]

            anchor_path, positive_path = random.sample(
                twin1_imgs, 2
            )

            negative_path = random.choice(
                twin2_imgs
            )

        # ===== حالة الأشخاص العاديين =====
        else:
            anchor_person = random.choice(
                self.people_keys
            )

            negative_person = random.choice([
                p for p in self.people_keys
                if p != anchor_person
            ])

            anchor_imgs = self.normal_people[
                anchor_person
            ]

            negative_imgs = self.normal_people[
                negative_person
            ]

            anchor_path, positive_path = random.sample(
                anchor_imgs, 2
            )

            negative_path = random.choice(
                negative_imgs
            )

        anchor = self.load_image(anchor_path)
        positive = self.load_image(positive_path)
        negative = self.load_image(negative_path)

        return anchor, positive, negative