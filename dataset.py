import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd


class AimonkDataset(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform

        # Prevent pandas from converting "NA" to NaN
        data = pd.read_csv(
            label_file,
            sep=r"\s+",
            header=None,
            keep_default_na=False
        )

        valid_rows = []

        for i in range(len(data)):
            image_name = data.iloc[i, 0]
            image_path = os.path.join(image_folder, image_name)

            if os.path.exists(image_path):
                valid_rows.append(data.iloc[i])

        self.data = pd.DataFrame(valid_rows).reset_index(drop=True)

        self.image_names = self.data.iloc[:, 0]
        self.labels = self.data.iloc[:, 1:]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_names.iloc[idx])
        image = Image.open(image_path).convert("RGB")

        label_row = self.labels.iloc[idx].values

        labels = []
        mask = []

        for val in label_row:
            if val == "NA":
                labels.append(0.0)
                mask.append(0.0)
            else:
                labels.append(float(val))
                mask.append(1.0)

        labels = torch.tensor(labels, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels, mask