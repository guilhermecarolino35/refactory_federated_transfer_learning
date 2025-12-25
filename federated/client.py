from federated.param_handler import set_parameters,get_parameters_from_net
from flwr.client import Client, ClientApp, NumPyClient
from trainer.test import test
from trainer.train import train
from federated.transferability.compute_kl import compute_kl_divergence
import torch
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== CLIENT ==========
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader,metrics_logger,transfer_manager,round_number=0):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.round = round_number
        self.frozen = True
        self.metrics_logger = metrics_logger
        self.global_net = copy.deepcopy(net)
        self.transfer_manager = transfer_manager
    def get_parameters(self, config):
        print(f"[Client {self.partition_id}] get_parameters")
        return get_parameters_from_net(self.net)

    def fit(self, parameters, config):
        self.round += 1 
        print(f"[Client {self.partition_id}] Iniciando round {self.round}")
        # Atualiza modelo local e global
        set_parameters(self.net, parameters)
        set_parameters(self.global_net, parameters)

        #antes do treinamento
        self.transfer_manager.on_round_start(
            self.global_net,
            self.net,
            self.trainloader,
            self.round
        )
        #treino
        train(self.net,self.trainloader,device,epochs=10)

        #Chamada depois do treinamento 
        transfer_metrics = self.transfer_manager.on_round_end(
            self.global_net,
            self.net,
            self.trainloader,
            self.round
        )

        return (
            get_parameters_from_net(self.net),
            len(self.trainloader),
            {
                "client_id": self.partition_id,
                "kl_transfer": transfer_metrics["kl_transfer"]
            }
            
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        print("chamou o evaluate do client")
        loss, accuracy = test(self.net, self.valloader,device)
        #self.metrics_logger.log_client_accuracy(client_id=self.partition_id,round_number=self.round,accuracy=accuracy)
        print(f"[Client {self.partition_id}] evaluate, config: {config}, loss: {loss}, accuracy: {accuracy} evaluate called")
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "round": int(self.round),"client_id": self.partition_id}