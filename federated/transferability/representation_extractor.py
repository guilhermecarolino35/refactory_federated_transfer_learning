import torch
from typing import Dict


class RepresentationExtractor:
    def __init__(self, device):
        self.device = device

    def extract(self, model, dataloader):
        model.eval()

        all_logits = []
        all_features = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["img"].to(self.device)

                
                features = model.extract_features(images)

                
                logits = model(images)

                all_features.append(features.detach().cpu())
                all_logits.append(logits.detach().cpu())

        return {
            "features": torch.cat(all_features, dim=0),
            "logits": torch.cat(all_logits, dim=0)
        }