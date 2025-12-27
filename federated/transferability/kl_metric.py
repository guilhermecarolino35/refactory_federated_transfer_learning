import torch.nn.functional as F # type: ignore
from federated.transferability.metric_base import TransferMetric


class KlTransferMetric(TransferMetric):
    def __init__(self,temperature: float =2.0):
       
       self.temperature = temperature

    def compute(self, repr_cache:dict, round_number:int) -> dict:
        logits_g = repr_cache["global"]["logits"]
        logits_l = repr_cache["local"]["logits"]
       
        p_g = F.softmax(logits_g /self.temperature, dim=1)
        log_p_l = F.log_softmax(logits_l/self.temperature, dim =1)
        kl_per_sample = F.kl_div(
            log_p_l, p_g, reduction="none"
        ).sum(dim=1)
        
        return {
            "kl_transfer": kl_per_sample.mean().item()
        }
