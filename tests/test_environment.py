"""Unit and integration tests for the Phase 1 federated learning environment."""

from collections import OrderedDict
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, FashionMNIST, MNIST
from torch.utils.data import TensorDataset
import flwr as fl
from flwr.clientapp import ClientApp
from flwr.serverapp import ServerApp
from flwr.server import ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr.common import ndarrays_to_parameters

from fedmaq.core.models import (
    SimpleCNN,
    ResNet18GN,
    get_model,
    get_model_parameters,
    set_model_parameters,
)
from fedmaq.core.partitioning import (
    generate_partition_indices,
    get_client_loader,
    get_server_loaders,
    load_dataset,
)
from fedmaq.core.telemetry import TelemetryManager
from fedmaq.core.strategy import TelemetryFedAvg
from fedmaq.core.client import GenericClient, LossHook, CompressionHook


@pytest.fixture
def mock_dataset(monkeypatch):
    """Fixture to mock torchvision dataset download and loading."""
    # Create 100 samples of 1-channel 28x28 images (MNIST-like)
    mock_data = torch.randn(100, 1, 28, 28)
    mock_labels = torch.randint(0, 10, (100,))
    mock_ds = TensorDataset(mock_data, mock_labels)
    # Add targets attribute for target extraction helper
    mock_ds.targets = mock_labels

    # Patch load_dataset to return the mock dataset
    monkeypatch.setattr(
        "fedmaq.core.partitioning.load_dataset", lambda name, train=True: mock_ds
    )
    return mock_ds


def test_model_factory_and_parameters():
    """Test get_model factory and get/set parameter helpers."""
    model = get_model("mnist", num_classes=10)
    assert isinstance(model, SimpleCNN)

    cifar_model = get_model("cifar10", num_classes=10)
    assert isinstance(cifar_model, ResNet18GN)

    # Test parameter helpers
    params = get_model_parameters(model)
    assert isinstance(params, list)
    assert len(params) > 0
    assert isinstance(params[0], np.ndarray)

    # Modify parameters and load back
    new_params = [p * 2.0 for p in params]
    set_model_parameters(model, new_params)

    # Verify modification
    re_extracted = get_model_parameters(model)
    for p_new, p_re in zip(new_params, re_extracted):
        np.testing.assert_allclose(p_new, p_re, rtol=1e-5)


def test_deterministic_dirichlet_partitioning(mock_dataset, tmp_path, monkeypatch):
    """Test Dirichlet partitioning with public reserve and local caching."""
    # Patch CACHE_DIR to temp directory for testing
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    num_clients = 3
    alpha = 0.5
    num_public = 10
    seed = 42

    # First run (generates cache)
    pub_idx1, client_dict1 = generate_partition_indices(
        "mnist", num_clients, alpha, num_public, seed
    )

    assert len(pub_idx1) == num_public
    assert len(client_dict1) == num_clients
    total_client_samples = sum(len(indices) for indices in client_dict1.values())
    assert len(pub_idx1) + total_client_samples == 100

    # Second run (retrieves from cache)
    pub_idx2, client_dict2 = generate_partition_indices(
        "mnist", num_clients, alpha, num_public, seed
    )

    # Verify determinism and cache retrieval
    assert pub_idx1 == pub_idx2
    for k in client_dict1.keys():
        assert client_dict1[k] == client_dict2[k]


def test_client_server_loaders(mock_dataset):
    """Test retrieval of client and server PyTorch DataLoaders."""
    client_dict = {"0": [0, 1, 2], "1": [3, 4]}
    pub_idx = [5, 6, 7]

    train_loader = get_client_loader(
        "mnist", client_id=0, client_indices_dict=client_dict, batch_size=2, train=True
    )
    assert len(train_loader.dataset) == 3

    pub_loader, test_loader = get_server_loaders("mnist", pub_idx, batch_size=2)
    assert len(pub_loader.dataset) == 3
    assert len(test_loader.dataset) == 100  # patched test dataset has 100 samples


def test_simulation_dry_run(mock_dataset, tmp_path, monkeypatch):
    """Test 1-round CPU dry-run simulation of the client/server environment."""
    monkeypatch.setattr("fedmaq.core.partitioning.CACHE_DIR", tmp_path)

    # 1. Setup partitioning
    public_indices, client_indices_dict = generate_partition_indices(
        "mnist", num_clients=2, alpha=0.5, num_public_samples=10, seed=42
    )

    # 2. Config dict
    cfg_dict = {
        "num_clients": 2,
        "batch_size": 2,
        "local_epochs": 1,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "total_rounds": 1,
        "seed": 42,
        "client_fraction": 1.0,
        "heterogeneity": {
            "bandwidth": {"min_mbps": 5.0, "max_mbps": 20.0},
            "compute": {"min_samples_per_sec": 100.0, "max_samples_per_sec": 500.0},
        },
        "telemetry": {
            "wandb_enabled": False,
            "project": "test",
            "mode": "offline",
        },
        "algorithm": {"name": "fedavg"},
        "dataset": {"name": "mnist", "num_classes": 10},
    }

    # 3. Telemetry and components
    telemetry = TelemetryManager(cfg_dict)

    def client_fn(context):
        partition_id = context.node_config["partition-id"]
        train_loader = get_client_loader(
            "mnist", partition_id, client_indices_dict, batch_size=2, train=True
        )
        model = get_model("mnist", num_classes=10)
        return GenericClient(
            cid=str(partition_id),
            trainloader=train_loader,
            testloader=train_loader,
            model=model,
            loss_hook=LossHook(),
            compressor_hook=CompressionHook(),
            config=cfg_dict,
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)

    strategy = None

    def server_fn(context):
        nonlocal strategy
        global_model = get_model("mnist", num_classes=10)
        initial_parameters = ndarrays_to_parameters(get_model_parameters(global_model))
        _, test_loader = get_server_loaders("mnist", public_indices, batch_size=2)

        def evaluate_fn(server_round, parameters, config):
            return 0.5, {"accuracy": 0.9}

        strategy = TelemetryFedAvg(
            telemetry_manager=telemetry,
            config=cfg_dict,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=2,
            min_available_clients=2,
            evaluate_fn=evaluate_fn,
            initial_parameters=initial_parameters,
        )
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=1))

    server_app = ServerApp(server_fn=server_fn)

    # 4. Run simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=2,
    )

    # Check that telemetry recorded cumulative bytes and simulated time
    assert telemetry.cumulative_bytes > 0
    assert strategy.simulated_time > 0
