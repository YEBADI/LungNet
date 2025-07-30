# 🫁 LungNet: Deep Learning for Chest X-ray Classification

> A PyTorch-based framework for training, evaluating, and interpreting deep learning models on chest X-ray images. Built on the NIH ChestX-ray14 dataset, with integrated explainability (Grad-CAM), experiment tracking (MLflow), and Docker-based deployment.

---

## 🔬 Overview

**LungNet** is an end-to-end image classification pipeline for identifying thoracic diseases in chest X-rays. It uses the NIH ChestX-ray14 dataset and supports both multi-label and binary classification tasks. The project is designed for:

- ✅ Flexible model training using PyTorch
- 🔍 Explainable predictions via **Grad-CAM** and optional **SHAP**
- 📊 MLflow-based experiment logging
- 🐳 Docker-ready deployment (locally or on the cloud)
- 📈 Easily extensible for other public datasets or clinical deployments

---

## 📂 Dataset Used

**NIH ChestX-ray14**

- 📸 112,120 frontal chest X-rays
- 👥 30,805 unique patients
- 🩻 14 thoracic pathology labels (multi-label)
- ✅ “No Finding” class included for healthy controls

📥 [Download here](https://nihcc.app.box.com/v/ChestXray-NIHCC)

---

## 🧱 Project Structure

```
lungnet/
├── data/                   # Local mount (not stored in Docker)
├── notebooks/              # EDA, training analysis, Grad-CAM visualizations
├── src/
│   ├── data_utils.py       # Data loading, preprocessing, augmentations
│   ├── models.py           # CNN model definitions (e.g., ResNet50, DenseNet121)
│   ├── train.py            # Model training loop with MLflow logging
│   ├── evaluate.py         # Performance metrics (AUC, F1, Confusion Matrix)
│   ├── predict.py          # Inference logic
│   └── explain.py          # Grad-CAM & SHAP explainability
├── app/
│   └── main.py             # FastAPI server (optional)
├── mlruns/                 # MLflow run history
├── Dockerfile
├── run.sh
├── requirements.txt
└── README.md

```
---
## 🚀 Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/lungnet.git
cd lungnet
pip install -r requirements.txt
```

### 2. Train the Model

```bash
MODE=train bash run.sh
```

### 3. Run Inference

```bash
MODE=predict bash run.sh --image path/to/image.jpg
```

### 4. Launch FastAPI Service (Optional)

```bash
MODE=serve bash run.sh
```

### 5. View MLflow UI

```bash
MODE=mlflow bash run.sh
```

---

## 🔍 Supported Labels (NIH)

- Atelectasis  
- Cardiomegaly  
- Consolidation  
- Edema  
- Effusion  
- Emphysema  
- Fibrosis  
- Hernia  
- Infiltration  
- Mass  
- Nodule  
- Pleural Thickening  
- Pneumonia  
- Pneumothorax  
- **No Finding** (healthy baseline)

---

## 🔎 Explainability Example

> Grad-CAM applied to a pneumonia-predicted image.

![gradcam-example]([docs/pneumonia_gradcam.jpg](https://www.kaggleusercontent.com/kf/168162207/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0.._zVYSDnDltljn6NENk568w.UB2oRNc0vjxAfj8PTw8YQLpShi5IY_cOjaputEKJemQgC4ZlLmR1J1jnVwX2NXVJ3bOON3B8uFfD7Rmwm6Cy5iqPtRXNmLfrMsDQhY5GjGurX-ZpVOlPNhRLOt5RnVPLwpmzR7bRaueyV9ybk9E0KSsCwHkqFWsk3GUBGsfKOnOUs-FA2jfSk8aKNqjywO_3HU5TcCCNzbL2i_SOQO90130FfoX1yDc95JwJ5rS-eSe87X2dV7I1cMFEGyG1cr8zJrAqRYy6pEPiQvb6r8e8iBYNoKjrk1KxZaGQuB6q55LaFcUl5oIghGUeoarV5OO78fxH2Kpu6EjcbzOa5DcMsMtOvFCbP-rHxDAM3RkIBOvtslxjXs62U17uwHzymM78WOIUcxdlKElIpcjN6kglf3TowxUIEYtmQsE-Bn8s5K9hhNgNvpbogVjiB0JUaSG-ns6TttksDFDg4SLbdu7GvQC5VqSDRL0-TSUjf9gLVWadKHHYYxH1cb_Y3__9HGdYKma-WjOzNUo-Vx-H6qxD_bqCa8KZx3rYD73pF8x5CqMHxVQDj5eRrnlhj7t2Mt8sU1DHU_S2K1iQ5ZWg6iPLsLioleC6YZZqie7u4D0BlvYaOZOsWNpk81zjEJt8JQB2TmbZ-iZxip18qK6mdceCTHPpTKes-tDZEGKMzPpfaSM.ETi9lJ--emtbztR-nMKa5Q/__results___files/__results___27_0.png))

---

## 🧠 Key Features

- 🧠 **Model Architecture**: ResNet, DenseNet, EfficientNet (drop-in support)  
- 🏷 **Multi-label** classification using sigmoid + binary cross entropy  
- ⚖️ **Class balancing**, oversampling, and weighted loss options  
- 🔬 **Grad-CAM** overlays to visualize influential regions  
- 📊 **MLflow integration** for experiment tracking  
- 🐳 **Dockerized** for reproducible training and inference  
- ✅ Compatible with both CPU and GPU environments  

---

## 📈 Metrics

- AUC (per class)  
- Micro/macro F1-score  
- Confusion Matrix (optional thresholding)  
- Calibration plots (future)  

---

## 📄 License

MIT License

---

## ✨ Future Work

- [ ] Integrated SHAP for pixel attribution  
- [ ] Support for EfficientNetV2 and Vision Transformers  
- [ ] Training on other datasets (e.g. CheXpert, RSNA Pneumonia)  
- [ ] Multi-modal fusion with patient metadata (e.g. age, sex)
