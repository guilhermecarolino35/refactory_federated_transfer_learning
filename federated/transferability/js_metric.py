import torch.nn.functional as F
from federated.transferability.metric_base import TransferMetric

class JSTransferMetric(TransferMetric):
    def __init__(self,temperature: float = 2.0):
        self.temperature = temperature

    def compute(self, repr_cache, round_number):
        logits_g = repr_cache["global"]["logits"]
        logits_l = repr_cache["local"]["logits"]

        #distribuicoes
        p_g = F.softmax(logits_g/self.temperature, dim=1)
        p_l = F.softmax(logits_l/self.temperature, dim=1)

        # distribuicao media

        m = 0.5 * (p_g + p_l)

        #KL(p_g || m)

        kl_gm = F.kl_div(
            F.log_softmax(logits_g/self.temperature, dim=1),m,reduction="none").sum(dim=1)

        #KL(p_l || m)

        kl_lm = F.kl_div(
            F.log_softmax(logits_l / self.temperature, dim=1),
            m,
            reduction="none"
        ).sum(dim=1)


        js_per_sample = 0.5 * (kl_gm + kl_lm)


        return {
            "js_transfer": js_per_sample.mean().item()
        }