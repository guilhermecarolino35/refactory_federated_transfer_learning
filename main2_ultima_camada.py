#torch
import torch 
import torch.nn as nn
import torch.nn.functional as F 

#src
#Models
from models.resnet18 import Net
from models.light_cnn import LightCNN
from federated.data.partitioner import load_client_datasets
from trainer.train import train 
from trainer.test import test
#flwr
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Metrics, Context
#Client
from federated.client_head_acuraccy import FlowerClient
#server 
from federated.server_head_acuraccy import make_server_fn
from flwr.server import ServerApp
#simulation
from flwr.simulation import run_simulation
from utils.metrics_logger import MetricsLogger
#Transferability manager



import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"



NUM_PARTITIONS = 10
EXPERIMENT_ID = 1

#Primeiro experimento numeros de clientes 10, 25, 50 . Porem todos com 10 rounds 

def get_alpha_for_run(run_id: int) -> float:
    alphas = {
        0: 0.05,
        1: 0.2,
        2: 0.5,
        3: 1.0,
        4: 3.0,
       
    }
    return alphas[run_id]

device = "cuda" if torch.cuda.is_available() else "cpu"




def make_client_fn(metrics_logger,alpha:float):
    def client_fn(context: Context) -> Client:
        net = LightCNN().to(device)

        partition_id = context.node_config["partition-id"]
        

        trainloader, valloader= load_client_datasets(
            partition_id=partition_id,
            num_partitions=NUM_PARTITIONS,
            alpha=alpha
        )
        return FlowerClient(
            partition_id,
            net,
            trainloader,
            valloader,
            metrics_logger,
        ).to_client()

    return client_fn



def run_experiment(experiment_id: int, experiment_run: int):


    alpha = get_alpha_for_run(experiment_run)

    metrics_logger = MetricsLogger(
        experiment_id=experiment_id,
        experiment_run=experiment_run,
    )

    client = ClientApp(
        client_fn=make_client_fn(metrics_logger, alpha)
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

NUM_RUNS = 5

for run_id in range(NUM_RUNS):
    run_experiment(EXPERIMENT_ID, run_id)