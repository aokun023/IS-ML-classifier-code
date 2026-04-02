"""Training utilities for IS-ML classification experiments."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    """Collect ground-truth and predicted labels."""

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Collecting predictions", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy().tolist()
            y_pred.extend(preds)
            y_true.extend(labels.numpy().tolist())
    return y_true, y_pred


class ClassificationTrainer:
    """Minimal trainer with best-validation checkpoint selection."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_path: Path,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

    def _run_epoch(self, loader: DataLoader, is_train: bool) -> tuple[float, float]:
        self.model.train(is_train)
        running_loss = 0.0
        correct = 0
        total = 0
        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for inputs, labels in tqdm(loader, desc="Train" if is_train else "Val", leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if is_train:
                    self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                running_loss += loss.item() * labels.size(0)
                correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
                total += labels.size(0)
        return running_loss / total, correct / total

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
    ) -> tuple[dict[str, list[float]], float]:
        """Train and return history together with elapsed time."""

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_acc = -1.0
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        for _ in tqdm(range(num_epochs), desc="Overall training"):
            train_loss, train_acc = self._run_epoch(train_loader, is_train=True)
            val_loss, val_acc = self._run_epoch(val_loader, is_train=False)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)

        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        return history, time.time() - start

    def test(self, loader: DataLoader) -> float:
        """Compute test accuracy."""

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Testing", leave=False):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                correct += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
                total += labels.size(0)
        return correct / total
