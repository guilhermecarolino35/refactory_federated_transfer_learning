#dirichlet_partitioner
from torch.utils.data import DataLoader
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
import torchvision.transforms as transforms

BATCH_SIZE = 64




def load_client_datasets(
    partition_id: int,
    num_partitions: int,
    alpha: float,
):
    # 1. Particionador Dirichlet (non-IID)
    dirichlet_partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        alpha=alpha,
        partition_by="label",
    )

    # 2. Dataset federado (somente treino é particionado)
    fds = FederatedDataset(
        dataset="cifar10",
        partitioners={"train": dirichlet_partitioner},
    )

    # 3. Carrega partição do cliente
    partition = fds.load_partition(partition_id)

    # 4. Split local: train / val
    partition_train_test = partition.train_test_split(
        test_size=0.2,
        seed=42,
    )

    # 5. Transforms
    pytorch_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # 6. DataLoaders locais
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valloader = DataLoader(
        partition_train_test["test"],
        batch_size=BATCH_SIZE,
    )

    return trainloader, valloader



def load_global_testloader():
    # 1. Dataset SEM particionamento
    fds = FederatedDataset(dataset="cifar10",partitioners={})

    # 2. Mesmo preprocessing dos clientes
    pytorch_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ])

    def apply_transforms(batch):
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    # 3. Teste global
    testset = fds.load_split("test").with_transform(apply_transforms)

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return testloader
