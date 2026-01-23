import torch
from torchvision import models
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def extract_features(self, x:torch.Tensor) -> torch.Tensor:
        
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)   # (B, 512, 1, 1)
        x = torch.flatten(x, 1)     # (B, 512)
        return x



    def freeze_backbone(self):
        """Freeze all layers except the classification head (fc)."""
        for name, param in self.model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False

        for param in self.model.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze the whole network."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.model.fc(features)
        return logits
    
