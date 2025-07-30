import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from src.data_utils import ChestXrayDataset
import mlflow.pytorch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Updated dataset path
dataset = ChestXrayDataset(
    csv_path='data/train.csv',
    images_dir='data/raw/'
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Simple ResNet18-based model
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(dataset.label_map))  # multi-label output
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

mlflow.set_experiment("lungnet-quick")
with mlflow.start_run():
    for epoch in range(1):  # just 1 epoch for quick test
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        mlflow.log_metric("loss", avg_loss)
    mlflow.pytorch.log_model(model, "model")
