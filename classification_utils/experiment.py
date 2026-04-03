"""High-level classification experiment runner."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader

from .datasets import (
    ACFTransform,
    NormalizeTensor,
    SpeckleGeneratedDataset,
    SpeckleRealDataset,
    compute_dataset_stats,
    load_generated_metadata,
)
from .models import MODEL_REGISTRY
from .training import ClassificationTrainer, collect_predictions, count_parameters, set_seed


def _extract_dataset_parameter(dataset_name: str, key: str) -> str:
    """Extract one parameter value from a dataset directory name."""

    match = re.search(rf"{re.escape(key)}-([^_]+)", dataset_name)
    return match.group(1) if match is not None else "unknown"


@dataclass
class ClassificationConfig:
    """Configuration for one classification study."""

    repo_root: Path
    real_data_root: Path
    real_metadata_filename: str = "metadata_256.csv"
    final_size: tuple[int, int] = (64, 64)
    canvas_size: tuple[int, int] = (256, 256)
    max_real_samples_per_class: int = 25
    max_gen_samples_per_class: int = 0
    use_acf: bool = False
    padding_mode: str = "zeros"
    shift_amount: int = 0
    shift_mode: str = "fixed"
    eval_protocol: str = "center_test"
    batch_size: int = 32
    num_epochs: int = 500
    random_seed: int = 42
    save_confusion_matrices: bool = False
    model_names: tuple[str, ...] = ("SimpleCNN", "ResNet18")
    learning_rates: dict[str, float] = field(
        default_factory=lambda: {"SimpleCNN": 6.8e-4, "ResNet18": 6.8e-4}
    )
    generated_data_root: Path | None = None
    generated_stage_folder: str = "stage5_pretrained_data"

    @property
    def real_metadata_path(self) -> Path:
        return self.real_data_root / self.real_metadata_filename

    @property
    def results_dir(self) -> Path:
        dataset_name = self.real_data_root.name
        z_value = _extract_dataset_parameter(dataset_name, "z")
        sigma_value = _extract_dataset_parameter(dataset_name, "sigma")
        input_tag = "acf" if self.use_acf else "intensity"
        dirname = (
            "classification_"
            f"z-{z_value}_"
            f"sigma-{sigma_value}_"
            f"crop-{self.final_size[0]}x{self.final_size[1]}_"
            f"input-{input_tag}_"
            f"pad-{self.padding_mode}_"
            f"shift-{self.shift_amount}-{self.shift_mode}_"
            f"eval-{self.eval_protocol}_"
            f"seed-{self.random_seed}"
        )
        return self.repo_root / "data" / "processed" / dirname


def _plot_confusion_matrix(y_true: list[int], y_pred: list[int], save_path: Path) -> None:
    """Save a normalized confusion matrix."""

    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix.astype(np.float64) / (matrix.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(15))
    ax.set_yticks(range(15))
    ax.set_xticklabels([format(i, "04b") for i in range(1, 16)], rotation=45, ha="right")
    ax.set_yticklabels([format(i, "04b") for i in range(1, 16)])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def run_classification_experiment(config: ClassificationConfig) -> pd.DataFrame:
    """Run the configured classification study and save results."""

    set_seed(config.random_seed)
    real_df = pd.read_csv(config.real_metadata_path)
    train_real, test_real = train_test_split(
        real_df,
        test_size=0.3,
        random_state=42,
        stratify=real_df["class_label"],
    )
    train_real, val_real = train_test_split(
        train_real,
        test_size=2 / 7,
        random_state=42,
        stratify=train_real["class_label"],
    )
    if config.max_real_samples_per_class > 0:
        train_real = train_real.groupby("class_label", group_keys=False).head(config.max_real_samples_per_class).reset_index(drop=True)

    gen_df = pd.DataFrame(columns=["filepath", "class_label"])
    if config.generated_data_root is not None and config.max_gen_samples_per_class > 0:
        gen_df = load_generated_metadata(config.generated_data_root, config.generated_stage_folder)
        if not gen_df.empty:
            min_count = int(gen_df["class_label"].value_counts().min())
            target = min(config.max_gen_samples_per_class, min_count)
            gen_df = gen_df.groupby("class_label", group_keys=False).head(target).reset_index(drop=True)

    mean, std = compute_dataset_stats(
        train_real,
        config.real_data_root,
        gen_df if not gen_df.empty else None,
        config.generated_data_root,
        config.canvas_size,
        config.final_size,
        config.shift_amount,
        config.shift_mode,
        config.use_acf,
        batch_size=config.batch_size,
    )
    transform_steps = []
    if config.use_acf:
        transform_steps.append(ACFTransform())
    transform_steps.append(NormalizeTensor(mean, std))

    def _apply_transform(tensor: torch.Tensor) -> torch.Tensor:
        for transform in transform_steps:
            tensor = transform(tensor)
        return tensor

    train_real_ds = SpeckleRealDataset(
        train_real,
        config.real_data_root,
        transform=_apply_transform,
        canvas_size=config.canvas_size,
        final_size=config.final_size,
        shift=config.shift_amount,
        shift_mode=config.shift_mode,
        is_train=True,
    )
    train_dataset = train_real_ds
    if not gen_df.empty and config.generated_data_root is not None:
        train_gen_ds = SpeckleGeneratedDataset(
            gen_df,
            config.generated_data_root,
            transform=_apply_transform,
            canvas_size=config.canvas_size,
            final_size=config.final_size,
            shift=config.shift_amount,
            shift_mode=config.shift_mode,
            is_train=True,
            stage_folder=config.generated_stage_folder,
        )
        train_dataset = ConcatDataset([train_real_ds, train_gen_ds])

    if config.eval_protocol == "center_test":
        val_dataset = SpeckleRealDataset(
            val_real,
            config.real_data_root,
            transform=_apply_transform,
            canvas_size=config.canvas_size,
            final_size=config.final_size,
            shift=config.shift_amount,
            shift_mode=config.shift_mode,
            is_train=False,
            apply_shift_in_eval=True,
        )
        test_dataset = SpeckleRealDataset(
            test_real,
            config.real_data_root,
            transform=_apply_transform,
            canvas_size=config.canvas_size,
            final_size=config.final_size,
            shift=0,
            shift_mode="random",
            is_train=False,
            apply_shift_in_eval=False,
        )
    else:
        raise ValueError(f"Unsupported eval protocol: {config.eval_protocol}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model_name in config.model_names:
        model = MODEL_REGISTRY[model_name](num_classes=15, in_channels=1, padding_mode=config.padding_mode)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rates[model_name], weight_decay=1e-5)
        trainer = ClassificationTrainer(
            model=model,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            checkpoint_path=config.results_dir / "checkpoints" / f"{model_name}_best.pt",
        )
        history, duration = trainer.train(train_loader, val_loader, num_epochs=config.num_epochs)
        test_acc = trainer.test(test_loader)
        (config.results_dir / f"history_{model_name}.json").write_text(json.dumps(history, indent=2))
        row = {
            "model": model_name,
            "use_acf": config.use_acf,
            "padding_mode": config.padding_mode,
            "shift_amount": config.shift_amount,
            "shift_mode": config.shift_mode,
            "test_accuracy": test_acc,
            "duration": duration,
            "num_parameters": count_parameters(trainer.model),
        }
        rows.append(row)

        if config.save_confusion_matrices:
            y_true, y_pred = collect_predictions(trainer.model, test_loader, trainer.device)
            _plot_confusion_matrix(y_true, y_pred, config.results_dir / f"confusion_matrix_{model_name}.png")

    results = pd.DataFrame(rows)
    results.to_csv(config.results_dir / "summary.csv", index=False)
    return results
