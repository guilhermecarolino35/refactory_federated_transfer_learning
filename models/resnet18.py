import torch
from torchvision import models
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # # Congela todas as camadas
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # Substitui a última camada (fc) e garante que ela esteja treinável
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        #self.fc = self.model.fc  # Alias direto

        # Garante que a nova fc tenha requires_grad=True
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
