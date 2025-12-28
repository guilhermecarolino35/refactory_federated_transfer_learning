from typing import List, Tuple, Dict
from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import ndarrays_to_parameters, NDArrays, Scalar, Context
import torch

from trainer.test import test
from federated.param_handler import set_parameters,get_parameters_from_net
from federated.data.partitioner import load_global_testloader

from models.resnet18 import Net


device = "cuda" if torch.cuda.is_available() else "cpu"

def make_weighted_average(metrics_logger):
    def weighted_average(metrics):
        """
        metrics: List[Tuple[int, Metrics]]
        """
        round_number = metrics_logger.current_round 
        total_examples = 0
        weighted_acc_sum = 0.0

        for num_examples, m in metrics:
            acc = m["accuracy"]
            #round_number = m["round"]
            client_id = m["client_id"]

            # ✅ LOG CENTRALIZADO NO SERVIDOR
            metrics_logger.log_client_accuracy(
                client_id=client_id,
                round_number=round_number,
                accuracy=acc,
            )

            weighted_acc_sum += num_examples * acc
            total_examples += num_examples

        global_acc = weighted_acc_sum / total_examples
        return {"accuracy": global_acc}

    return weighted_average






# FUNÇÃO: Avaliação centralizada no servidor
def get_evaluate_fn(metrics_logger):
    """Return an evaluation function for server-side evaluation."""
    def evaluate_fn(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
        metrics_logger.current_round = server_round 
        # Carrega o modelo global no servidor
        net = Net().to(device)
        set_parameters(net, parameters)
        print(f">>> Server evaluate for round {server_round} started teste do evaluate")
        # CARREGA O TESTLOADER GLOBAL (não particionado)
        # Usamos qualquer partition_id apenas para acessar a função de carregamento
        testloader_global = load_global_testloader()

        # Avalia no conjunto de teste global
        loss, accuracy = test(net, testloader_global,device)
        # LOG global por round
        metrics_logger.log_global_accuracy(round_number=server_round,accuracy=accuracy)
        print(f" [Servidor] Round {server_round}: Teste Global - loss {loss:.4f}, accuracy {accuracy:.4f}")

        return loss, {"global_accuracy": accuracy}

    return evaluate_fn




def make_fit_metrics_aggregation_fn(metrics_logger):

    def aggregate_fit_metrics(metrics):
        """
        metrics: List[Tuple[int, Dict[str, Scalar]]]
        """

        if not metrics:
            return {}

        round_number = metrics_logger.current_round

        weighted_kl_sum = 0.0
        total_examples = 0

        for num_examples, m in metrics:

            client_id = m["client_id"]
            kl_value = m["kl_transfer"]

            # média ponderada
            weighted_kl_sum += num_examples * kl_value
            total_examples += num_examples

            # log por cliente
            metrics_logger.log_client_kl(
                client_id=client_id,
                round_number=round_number,
                kl=kl_value
            )

        aggregated_metrics = {}

        # agregação global do round
        if total_examples > 0:
            kl_mean = weighted_kl_sum / total_examples
            aggregated_metrics["kl_transfer_mean"] = kl_mean

            metrics_logger.log_global_kl(
                round_number=round_number,
                kl_mean=kl_mean
            )

        return aggregated_metrics

    return aggregate_fit_metrics

#factory
def make_server_fn(metrics_logger):
    # server_fn com avaliação centralizada
    def server_fn(context: Context) -> ServerAppComponents:
        """Construct components that set the ServerApp behaviour."""
        net = Net().to(device)
        params = get_parameters_from_net(net)

        # Create FedAvg strategy with CENTRALIZED evaluation
        strategy = FedAvg(
            fraction_fit=1.0,  # Sample 100% of available clients for training
            fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
            min_fit_clients=10,  # Never sample less than 10 clients for training
            min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
            min_available_clients=10,  # Wait until all 10 clients are available
            initial_parameters=ndarrays_to_parameters(params),
            fit_metrics_aggregation_fn = make_fit_metrics_aggregation_fn(metrics_logger),
            evaluate_metrics_aggregation_fn=make_weighted_average(metrics_logger),  # Aggregate client metrics
            # Avaliação centralizada no servidor
            evaluate_fn=get_evaluate_fn(metrics_logger),  # Centralized evaluation on global test set
        )

        # Configure the server for 5 rounds of training
        config = ServerConfig(num_rounds=2)

        return ServerAppComponents(strategy=strategy, config=config)

    return server_fn


