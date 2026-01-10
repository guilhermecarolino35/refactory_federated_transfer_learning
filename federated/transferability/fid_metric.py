import torch
import torch.nn.functional as F
from federated.transferability.metric_base import TransferMetric


class FrechetTransferMetric(TransferMetric):
    """
    Fréchet Distance (FID-style) entre distribuições de features globais e locais.
    Implementação numericamente estável para N << D.
    """

    def __init__(self, eps: float = 1e-6, normalize: bool = True, use_double: bool = True):
        self.eps = eps
        self.normalize = normalize
        self.use_double = use_double

    def compute(self, repr_cache: dict, round_number: int) -> dict:
        X = repr_cache["global"]["features"]  # (N, D)
        Y = repr_cache["local"]["features"]   # (N, D)

        # (opcional, mas recomendado)
        if self.use_double:
            X = X.double()
            Y = Y.double()

        # normalização L2 (opcional)
        if self.normalize:
            X = F.normalize(X, dim=1)
            Y = F.normalize(Y, dim=1)

        # médias
        mu_x = X.mean(dim=0)
        mu_y = Y.mean(dim=0)

        # covariâncias
        sigma_x = self._covariance(X, mu_x)
        sigma_y = self._covariance(Y, mu_y)

        frechet = self._frechet_distance(mu_x, mu_y, sigma_x, sigma_y)

        return {
            "fid_transfer": float(frechet.item())
        }

    def _covariance(self, X: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Σ = (1 / (N - 1)) * (X - μ)^T (X - μ)
        """
        X_centered = X - mu
        return (X_centered.T @ X_centered) / (X.shape[0] - 1)

    def _frechet_distance(
        self,
        mu_x: torch.Tensor,
        mu_y: torch.Tensor,
        sigma_x: torch.Tensor,
        sigma_y: torch.Tensor,
    ) -> torch.Tensor:
        """
        ||μ_x - μ_y||² + Tr(Σ_x + Σ_y - 2(Σ_x Σ_y)^{1/2})
        """

        # termo da diferença das médias
        diff = mu_x - mu_y
        diff_term = diff @ diff

        # regularização numérica
        eye = torch.eye(sigma_x.shape[0], device=sigma_x.device, dtype=sigma_x.dtype)
        sigma_x = sigma_x + self.eps * eye
        sigma_y = sigma_y + self.eps * eye

        # produto simetrizado (IMPORTANTE)
        A = sigma_x @ sigma_y
        A = (A + A.T) / 2.0

        # raiz de matriz
        covmean = self._matrix_sqrt(A)

        trace_term = torch.trace(sigma_x + sigma_y - 2.0 * covmean)

        frechet = diff_term + trace_term

        # 🔒 clamp final (FID nunca pode ser negativa)
        frechet = torch.clamp(frechet, min=0.0)

        return frechet

    def _matrix_sqrt(self, A: torch.Tensor) -> torch.Tensor:
        """
        Raiz de matriz simétrica positiva definida via decomposição espectral.
        """
        eigvals, eigvecs = torch.linalg.eigh(A)

        # remove autovalores negativos numéricos
        eigvals = torch.clamp(eigvals, min=0.0)

        sqrt_eigvals = torch.sqrt(eigvals)

        return eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T