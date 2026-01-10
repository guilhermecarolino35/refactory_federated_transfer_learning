from federated.param_handler import set_parameters,get_parameters_from_net
from flwr.client import Client, ClientApp, NumPyClient
from trainer.test import test
from trainer.train import train
import torch
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== CLIENT ==========
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader,metrics_logger,transfer_manager,repr_extractor,round_number=0):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = round_number
        self.frozen = True
        self.metrics_logger = metrics_logger
        self.global_net = copy.deepcopy(net)
        self.transfer_manager = transfer_manager
        
        self.repr_extractor = repr_extractor
    
    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters_from_net(self.net)

    def fit(self, parameters, config):
        self.round += 1 
        # Atualiza modelo local e global
        set_parameters(self.net, parameters)
        set_parameters(self.global_net, parameters)

        global_repr = self.repr_extractor.extract(
            self.global_net,
            self.valloader
        )

        
        #treino
        train(self.net,self.trainloader,device,epochs=10)

       
        local_repr = self.repr_extractor.extract(
            self.net,
            self.valloader
        )

        repr_cache = {
            "global":global_repr,
            "local":local_repr
        }


        transfer_metrics = self.transfer_manager.compute_all(repr_cache,self.round)

        return (
            get_parameters_from_net(self.net),
            len(self.trainloader),
            {
                "client_id": self.partition_id,
                "kl_transfer" : transfer_metrics["kl_transfer"],
                "js_transfer" : transfer_metrics["js_transfer"],
                "mmd_transfer" : transfer_metrics["mmd_transfer"],
                "frechet_transfer": transfer_metrics["fid_transfer"],
            }
            
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader,device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "round": int(self.round),"client_id": self.partition_id}