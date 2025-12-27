import torch
import torch.nn.functional as F

def compute_js(
    global_net,
    local_net,
    dataloader,
    device,
    temperature: float = 2.0,
):

