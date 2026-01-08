import torch.nn.functional as F
from federated.transferability.metric_base import TransferMetric
import torch

class MMDTransferMetric(TransferMetric):
     def __init__(self,sigma:float = 1.0, normalize:bool =True):
         """
         sigma: largura do kernel RBF
         normalize: se True, normaliza features 
         """
         self.sigma = sigma
         self.normalize = normalize


     def compute(self, repr_cache:dict, round_number:int) -> dict:
         features_g = repr_cache["global"]["features"]  # (N, D)
         features_l = repr_cache["local"]["features"]   # (N, D)

          # opcional: normalização L2
         if self.normalize:
             features_g = F.normalize(features_g, dim=1)
             features_l = F.normalize(features_l, dim=1)

          # cálculo da MMD²
         mmd_value = self._compute_mmd(features_g, features_l)

         return {
             "mmd_transfer":float (mmd_value.item())
         }
    
     def _rbf_kernel(self,X: torch.Tensor, Y:torch.Tensor):
         XX = (X ** 2).sum(dim=1, keepdim=True)
         YY = (Y ** 2).sum(dim=1, keepdim=True)
         distances = XX - 2 * X @ Y.T + YY.T
         return torch.exp(-distances / (2 * self.sigma ** 2))
    
     def _compute_mmd(self, X: torch.Tensor, Y: torch.Tensor):
         K_xx = self._rbf_kernel(X, X)
         K_yy = self._rbf_kernel(Y, Y)
         K_xy = self._rbf_kernel(X, Y)
         return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    


