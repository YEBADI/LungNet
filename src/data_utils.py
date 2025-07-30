import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, images_dir, transform=None, label_map=None):
        self.df = pd.read_csv(csv_path)

        # Handle multi-label case
        self.df['Finding Labels'] = self.df['Finding Labels'].str.split('|')

        # Build full image paths
        self.df['Image Path'] = self.df['Image Index'].apply(lambda x: os.path.join(images_dir, x))

        # Create sorted unique list of labels if not provided
        if label_map is None:
            all_labels = sorted({label for labels in self.df['Finding Labels'] for label in labels})
            self.label_map = {label: i for i, label in enumerate(all_labels)}
        else:
            self.label_map = label_map

        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['Image Path']).convert("RGB")
        img = self.transform(img)

        # Multi-label target vector
        target = torch.zeros(len(self.label_map))
        for label in row['Finding Labels']:
            if label in self.label_map:
                target[self.label_map[label]] = 1.0

        return img, target
