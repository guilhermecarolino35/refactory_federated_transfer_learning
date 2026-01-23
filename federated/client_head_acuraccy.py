from federated.param_handler import set_parameters,get_parameters_from_net
from flwr.client import Client, ClientApp, NumPyClient
from trainer.test import test
from trainer.train import train
import torch
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== CLIENT ==========
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader,metrics_logger,round_number=0):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = round_number
        self.frozen = True
        self.metrics_logger = metrics_logger

        

    


    
    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters_from_net(self.net)

    def fit(self, parameters, config):
        server_round = config["server_round"]

        if server_round == 1:
            self.net.freeze_backbone()
        else:
            self.net.unfreeze_backbone()

        # Atualiza modelo local e global
        set_parameters(self.net, parameters)
          
        #treino
        train(self.net,self.trainloader,device,epochs=10)
        loss,accuracy = test(self.net, self.valloader,device)
       

        return (
            get_parameters_from_net(self.net),
            len(self.trainloader),
            {
                "client_id": self.partition_id,
                "accuracy": accuracy
               
            }
            
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader,device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "round": int(self.round),"client_id": self.partition_id}