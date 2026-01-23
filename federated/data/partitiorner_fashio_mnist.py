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
        min_partition_size=1,
    )

    # 2. Dataset federado (somente treino é particionado)
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={"train": dirichlet_partitioner},
    )

    # 3. Carrega partição do cliente
    partition = fds.load_partition(partition_id)

    # 4. Split local: train / val
    partition_train_test = partition.train_test_split(
        test_size=0.2,
        seed=42,
    )

    # 5. Transforms (ResNet + ImageNet)
    pytorch_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5,),
        std=(0.5,),
    ),
    ])

    # 6. Adapter: HF -> contrato interno (img, label)
    def apply_transforms(batch):
        # HF usa "image"
        images = [pytorch_transforms(img) for img in batch["image"]]

       
        batch["img"] = images
        # remove chave HF para evitar bugs futuros
        del batch["image"]

        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)

    # 7. DataLoaders locais
    trainloader = DataLoader(
        partition_train_test["train"],
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    valloader = DataLoader(
        partition_train_test["test"],
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return trainloader, valloader


def load_global_testloader():
    # 1. Dataset global (SEM particionamento, mas argumento obrigatório)
    fds = FederatedDataset(
        dataset="zalando-datasets/fashion_mnist",
        partitioners={},   # <-- OBRIGATÓRIO na sua versão
    )

    # 2. MESMO preprocessing dos clientes
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),           # (1, 28, 28)
        transforms.Normalize(
            mean=(0.5,),
            std=(0.5,),
        ),
    ])

    # 3. Adapter HF -> contrato interno
    def apply_transforms(batch):
        images = [pytorch_transforms(img) for img in batch["image"]]
        batch["img"] = images
        del batch["image"]
        return batch

    # 4. Split de teste global
    testset = fds.load_split("test").with_transform(apply_transforms)

    testloader = DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return testloader