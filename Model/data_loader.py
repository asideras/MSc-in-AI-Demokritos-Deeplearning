import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class Data_loader(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, dtype={'id': str}).head(112)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        id = self.img_labels.iloc[idx, 0].lstrip('0')
        filename = f"{id}_second_out.jpg"
        img_path = os.path.join(self.img_dir, filename)
        image = read_image(img_path).float()

        if self.transform:
            image = self.transform(image)
        target = torch.tensor(self.img_labels.iloc[idx, 6:])

        return image, target.float()
