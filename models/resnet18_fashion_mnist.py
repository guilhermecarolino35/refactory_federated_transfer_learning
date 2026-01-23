import torch
import torch.nn as nn
from torchvision import models


class ResNet18FashionMNIST(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # 1️⃣ ResNet18 SEM pesos ImageNet
        self.model = models.resnet18(weights=None)

        # 2️⃣ Ajuste para imagens 1x28x28
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # 3️⃣ Remove downsampling agressivo
        self.model.maxpool = nn.Identity()

        # 4️⃣ Head de classificação
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    # ===============================
    # Feature extractor explícito
    # ===============================
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    # ===============================
    # Freeze / unfreeze (FL friendly)
    # ===============================
    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    # ===============================
    # Forward
    # ===============================
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        features = self.extract_features(x)
        logits = self.model.fc(features)
        return logits