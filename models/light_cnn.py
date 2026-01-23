import torch
import torch.nn as nn
import torch.nn.functional as F

class LightCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Linear(128, num_classes)

    
    def freeze_backbone(self):
        for param in self.features_extractor.parameters():
            param.requires_grad = False
        for param in self.gap.parameters():
            param.requires_grad = False

        for param in self.classifier.parameters():
            param.requires_grad = True
    
    
    def unfreeze_backbone(self):
        """Unfreeze the whole network."""
        for param in self.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_extractor(x)
        x = self.gap(x)            # (B, 128, 1, 1)
        x = torch.flatten(x, 1)    # (B, 128)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits