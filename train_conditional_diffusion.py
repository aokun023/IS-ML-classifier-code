#!/usr/bin/env python3
"""Thin entry point for conditional diffusion training and sample export."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator

from generation_utils import PipelineConfig, run_generation_pipeline, set_seed


def _resolve_repo_relative_path(repo_root: Path, path_like: str | Path) -> Path:
    """Resolve a repository-relative path from the configuration block."""

    path = Path(path_like)
    return path if path.is_absolute() else repo_root / path


CONFIG = {
    "dataset_root": "data/raw/dataset_z-5.00_sigma-5e-05",
    "metadata_filename": "metadata_256.csv",
    "dataset_tag": None,
    "input_resolution_tag": "256",
    "model_type": "Diffusion",
    "prediction_type": "v_prediction",
    "loss_target_type": "sample",
    "pixel_loss_type": "MSE",
    "lambda_freq": 1.0,
    "image_size": 256,
    "num_class_embed": 15,
    "random_seed": 42,
    "val_split_ratio": 0.3,
    "samples_per_class": 150,
    "training_samples_per_class": 105,
    "validation_samples_per_class": 45,
    "train_batch_size": 8,
    "val_batch_size": 24,
    "pretrain_epochs": 200,
    "pretrain_lr": 1e-4,
    "pretrain_lr_patience": 10,
    "pretrain_lr_factor": 0.5,
    "num_samples_to_generate": 50,
    "generation_batch_size": 5,
    "append_generated_samples": False,
    "class_ids": list(range(1, 16)),
    "model_config": {
        "sample_size": 256,
        "in_channels": 1,
        "out_channels": 1,
        "layers_per_block": 2,
        "block_out_channels": (128, 256, 384, 512, 512, 512),
        "down_block_types": (
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        "up_block_types": (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    },
}


def build_config() -> PipelineConfig:
    return PipelineConfig(
        repo_root=REPO_ROOT,
        dataset_path=_resolve_repo_relative_path(REPO_ROOT, CONFIG["dataset_root"]),
        metadata_filename=str(CONFIG["metadata_filename"]),
        dataset_tag=CONFIG["dataset_tag"],
        input_resolution_tag=str(CONFIG["input_resolution_tag"]),
        model_type=str(CONFIG["model_type"]),
        prediction_type=str(CONFIG["prediction_type"]),
        loss_target_type=str(CONFIG["loss_target_type"]),
        pixel_loss_type=str(CONFIG["pixel_loss_type"]),
        lambda_freq=float(CONFIG["lambda_freq"]),
        image_size=int(CONFIG["image_size"]),
        num_class_embed=int(CONFIG["num_class_embed"]),
        random_seed=int(CONFIG["random_seed"]),
        val_split_ratio=float(CONFIG["val_split_ratio"]),
        samples_per_class=int(CONFIG["samples_per_class"]),
        training_samples_per_class=int(CONFIG["training_samples_per_class"]),
        validation_samples_per_class=int(CONFIG["validation_samples_per_class"]),
        train_batch_size=int(CONFIG["train_batch_size"]),
        val_batch_size=int(CONFIG["val_batch_size"]),
        pretrain_epochs=int(CONFIG["pretrain_epochs"]),
        pretrain_lr=float(CONFIG["pretrain_lr"]),
        pretrain_lr_patience=int(CONFIG["pretrain_lr_patience"]),
        pretrain_lr_factor=float(CONFIG["pretrain_lr_factor"]),
        num_samples_to_generate=int(CONFIG["num_samples_to_generate"]),
        generation_batch_size=int(CONFIG["generation_batch_size"]),
        append_generated_samples=bool(CONFIG["append_generated_samples"]),
        class_ids=list(CONFIG["class_ids"]),
        model_config=dict(CONFIG["model_config"]),
    )


def main() -> None:
    config = build_config()
    set_seed(config.random_seed)
    accelerator = Accelerator(mixed_precision="bf16")
    run_generation_pipeline(config, accelerator)


if __name__ == "__main__":
    main()
