"""Generic Flower Client implementation with customizable hooks for loss and compression."""

from typing import Any, Dict, List, Tuple
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
from fedmaq.core.models import get_model_parameters, set_model_parameters


class LossHook:
    """Base class for customizing local training loss functions."""

    def on_train_begin(self, model: nn.Module) -> None:
        """Hook called before the first training batch."""
        pass

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute the local loss."""
        return criterion(outputs, targets)


class FedProxLossHook(LossHook):
    """Loss hook adding proximal L2 regularization for FedProx."""

    def __init__(self, mu: float = 0.01) -> None:
        self.mu = mu
        self.global_params: List[torch.Tensor] = []

    def on_train_begin(self, model: nn.Module) -> None:
        # Save a frozen copy of the initial global weights
        self.global_params = [
            p.clone().detach() for p in model.parameters() if p.requires_grad
        ]

    def compute_loss(
        self,
        model: nn.Module,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        loss = criterion(outputs, targets)
        proximal_term = 0.0
        params = [p for p in model.parameters() if p.requires_grad]
        for p, gp in zip(params, self.global_params):
            proximal_term += torch.sum((p - gp) ** 2)
        return loss + (self.mu / 2.0) * proximal_term


class CompressionHook:
    """Base class for compressing client model updates (deltas)."""

    def compress(self, deltas: List[np.ndarray]) -> Tuple[List[np.ndarray], int]:
        """Compress deltas and return (compressed_deltas, byte_size)."""
        # Default: Identity (uncompressed Float32 weights -> 4 bytes per element)
        byte_size = sum(d.nbytes for d in deltas)
        return deltas, byte_size


class GenericClient(fl.client.NumPyClient):
    """Extensible client wrapping a PyTorch model and executing local epochs."""

    def __init__(
        self,
        cid: str,
        trainloader: torch.utils.data.DataLoader,
        testloader: torch.utils.data.DataLoader,
        model: nn.Module,
        loss_hook: LossHook,
        compressor_hook: CompressionHook,
        config: Dict[str, Any],
    ) -> None:
        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.loss_hook = loss_hook
        self.compressor_hook = compressor_hook
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_properties(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"cid": self.cid}

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        # Load incoming server weights
        set_model_parameters(self.model, parameters)

        # Retrieve round configurations (with defaults from config)
        exp_config = self.config.get("experiment", self.config)
        lr = float(config.get("lr", exp_config.get("learning_rate", 0.01)))
        epochs = int(config.get("epochs", exp_config.get("local_epochs", 5)))
        weight_decay = float(exp_config.get("weight_decay", 0.0))

        # Setup training
        self.model.train()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        self.loss_hook.on_train_begin(self.model)

        for _ in range(epochs):
            for images, labels in self.trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_hook.compute_loss(
                    self.model, outputs, labels, criterion
                )
                loss.backward()
                optimizer.step()

        # Extract updated parameters
        updated_params = get_model_parameters(self.model)

        # Compute delta = w_new - w_old
        deltas = [u - o for u, o in zip(updated_params, parameters)]

        # Compress updates
        compressed_deltas, byte_size = self.compressor_hook.compress(deltas)

        # Reconstruct parameter update: w_new_reconstructed = w_old + compressed_deltas
        reconstructed_params = [o + cd for o, cd in zip(parameters, compressed_deltas)]

        return (
            reconstructed_params,
            len(self.trainloader.dataset),
            {"bytes_uploaded": byte_size, "partition_id": int(self.cid)},
        )

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Any]
    ) -> Tuple[float, int, Dict[str, Any]]:
        # Load weights
        set_model_parameters(self.model, parameters)

        self.model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss_sum += criterion(outputs, labels).item() * len(labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        loss = loss_sum / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0

        return float(loss), total, {"accuracy": float(accuracy)}
