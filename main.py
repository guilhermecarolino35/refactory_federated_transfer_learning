#torch
import torch 
import torch.nn as nn
import torch.nn.functional as F 

#src
from models.resnet18 import Net
from federated.data.partitioner import load_datasets
from trainer.train import train 
from trainer.test import test
#flwr
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
#Client
from federated.client import FlowerClient
#server 
from federated.server import make_server_fn
from flwr.server import ServerApp
#simulation
from flwr.simulation import run_simulation
from utils.metrics_logger import MetricsLogger
#Transferability manager
from federated.transferability.kl_metric import KlTransferMetric
from federated.transferability.manager import TransferabilityManager

NUM_PARTITIONS = 10
EXPERIMENT_ID = 7

device = "cuda" if torch.cuda.is_available() else "cpu"




def make_client_fn(metrics_logger):
    def client_fn(context: Context) -> Client:
        net = Net().to(device)

        partition_id = context.node_config["partition-id"]
        round_number = 0

        trainloader, valloader, _ = load_datasets(
            partition_id=partition_id
        )
        transfer_metrics = [
            KlTransferMetric(device=device)
        ]
        transfer_manager = TransferabilityManager(transfer_metrics)
        return FlowerClient(
            partition_id,
            net,
            trainloader,
            valloader,
            metrics_logger,
            transfer_manager,
        ).to_client()

    return client_fn



def run_experiment(experiment_id: int, experiment_run: int):

    metrics_logger = MetricsLogger(
        experiment_id=experiment_id,
        experiment_run=experiment_run,
    )

    client = ClientApp(
        client_fn=make_client_fn(metrics_logger)
    )

    server = ServerApp(
        server_fn=make_server_fn(metrics_logger)  # depois ligamos logger global
    )

    backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 0.0}}
    if device == "cuda":
        backend_config = {"client_resources": {"num_cpus": 1, "num_gpus": 1.0}}

    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )
    metrics_logger.save()

    print(
        f"✅ Experimento concluído | "
        f"id={experiment_id} | run={experiment_run}"
    )

NUM_RUNS = 1

for run_id in range(NUM_RUNS):
    run_experiment(EXPERIMENT_ID, run_id)