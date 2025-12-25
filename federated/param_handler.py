from collections import OrderedDict
from typing import List, Tuple
import numpy as np
import torch


# ========== PARAM HANDLING ==========
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict()

    for (name, _), param in zip(net.state_dict().items(), parameters):
        # Verifica se é um parâmetro numérico ou um buffer especial
        if param.size == 0 or param.dtype == np.dtype('O'):
            # Mantém o valor original da rede para buffers e parâmetros especiais
            state_dict[name] = net.state_dict()[name]
        else:
            # Converte normalmente para tensores PyTorch
            state_dict[name] = torch.from_numpy(param)

    net.load_state_dict(state_dict, strict=True)


def get_parameters_from_net(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
