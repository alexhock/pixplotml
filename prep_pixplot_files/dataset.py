import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class MetadataDataset(Dataset):
    def __init__(
        self, annotations_file_df, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = annotations_file_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx]["filename"])
        image = Image.open(img_path)

        label = self.img_labels.iloc[idx]["class_id"]
        if self.transform:
            # assumes albumentations
            image = self.transform(image=np.array(image))["image"]
        if self.target_transform:
            label = self.target_transform(label)

        return image, torch.tensor(label)


def collate_fn(batch):

    images, labels = zip(*batch)

    images = torch.stack(images)
    labels = torch.stack(labels)

    return images, labels
