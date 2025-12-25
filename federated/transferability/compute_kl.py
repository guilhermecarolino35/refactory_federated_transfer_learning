import torch
import torch.nn.functional as F

def compute_kl_divergence(
    global_net,
    local_net,
    dataloader,
    device,
    temperature: float = 2.0,
):
    """
    Calcula o KL médio entre o modelo global e o modelo local
    nos dados do cliente.

    Retorna:
        kl_mean (float): KL médio do cliente
    """
    total_kl  = 0.0
    total_samples = 0
    global_net.eval()
    local_net.eval()

    kl_values = []

    with torch.no_grad():
        for batch in dataloader:
            
            x = batch["img"].to(device)
            batch_size = x.size(0)
            # 1️⃣ Logits
            z_g = global_net(x)   # [B, C]
            z_k = local_net(x)    # [B, C]

            # 2️⃣ Softmax com temperatura
            q_g = F.softmax(z_g / temperature, dim=1) # global
            q_k = F.softmax(z_k / temperature, dim=1) # client 

            # 3️⃣ KL por amostra
            # F.kl_div espera log-probabilidades no primeiro argumento
            kl_batch = F.kl_div(
                q_k.log(),     # log q_k
                q_g,           # q_g
                reduction="none"
            ).sum(dim=1)       # soma sobre classes → KL_i

            total_kl += kl_batch.sum().item()
            total_samples += batch_size

    # 4️⃣ Média final KL_k
    kl_mean = total_kl/total_samples

    return kl_mean