from federated.transferability.metric_base import TransferMetric
from federated.transferability.compute_kl import compute_kl_divergence

class KlTransferMetric(TransferMetric):
    def __init__(self,device):
       self.device = device
       self.kl_value = None

    def on_round_start(self, global_model, local_model, dataloader, round_number):
       pass
    
    def on_round_end(self, global_model, local_model, dataloader, round_number):
       self.kl_value = compute_kl_divergence(global_model,local_model,dataloader,self.device)

    def compute(self) -> dict:
       return {
          "kl_transfer":float(self.kl_value)
          }