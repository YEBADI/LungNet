# src/data_utils.py
import pandas as pd, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        df = pd.read_csv(csv_file)
        self.df = df[df['Finding Labels'].notnull()]
        self.df['label'] = self.df['Finding Labels'].apply(lambda x: 0 if x == 'No Finding' else 1)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(f"{self.img_dir}/{row['Image Index']}").convert("RGB")
        label = torch.tensor(row['label'], dtype=torch.float32)
        return self.transform(image), label

