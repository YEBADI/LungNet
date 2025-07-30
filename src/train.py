# src/train.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch, torch.nn as nn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from src.data_utils import ChestXrayDataset
import mlflow.pytorch

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = ChestXrayDataset("Data_Entry_2017.csv", "images/")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

mlflow.set_experiment("lungnet-quick")
with mlflow.start_run():
    for epoch in range(1):
        for x, y in loader:
            x, y = x.to(device), y.unsqueeze(1).to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        mlflow.log_metric("loss", loss.item())
    mlflow.pytorch.log_model(model, "model")

