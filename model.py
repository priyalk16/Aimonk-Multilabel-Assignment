import torch.nn as nn
from torchvision import models


def get_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Train only final layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model