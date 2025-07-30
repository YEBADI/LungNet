# src/explain.py
from torchvision.models import resnet18
import cv2, numpy as np
import torch
import matplotlib.pyplot as plt

def gradcam(model, image_tensor, target_layer):
    gradients = []
    activations = []

    def forward_hook(module, input, output):
        activations.append(output)
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(image_tensor.unsqueeze(0))
    output[0][0].backward()

    act = activations[0].squeeze().detach().numpy()
    grad = gradients[0].squeeze().detach().numpy()
    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    h1.remove()
    h2.remove()
    return cam

