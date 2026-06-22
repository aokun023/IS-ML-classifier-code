#!/usr/bin/env python3
"""Run a paper-level IS-ML classification preset."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from classification_utils import ClassificationConfig, run_classification_experiment


def _resolve_repo_relative_path(repo_root: Path, path_like: str | Path | None) -> Path | None:
    """Resolve a repository-relative path from the configuration block."""

    if path_like is None:
        return None
    path = Path(path_like)
    return path if path.is_absolute() else repo_root / path


BASE_CONFIG = {
    "real_data_root": "data/raw/dataset_z-5.00_sigma-1.1_l0-1.5_Nx-2048",
    "real_metadata_filename": "metadata_256.csv",
    "final_size": (64, 64),
    "canvas_size": (256, 256),
    "max_real_samples_per_class": 50,
    "max_gen_samples_per_class": 0,
    "use_acf": False,
    "padding_mode": "zeros",
    "shift_amount": 0,
    "shift_mode": "fixed",
    "eval_protocol": "center_test",
    "batch_size": 32,
    "num_epochs": 500,
    "random_seed": 42,
    "save_confusion_matrices": False,
    "model_names": ("SimpleCNN", "ResNet18"),
    "learning_rates": {"SimpleCNN": 6.8e-4, "ResNet18": 6.8e-4},
    "use_lr_finder": True,
    "lr_finder_start_lr": 1e-7,
    "lr_finder_end_lr": 1e-2,
    "lr_finder_num_iter": 150,
    "generated_data_root": "results/generation_z-5.00_sigma-1.1_pred-v_prediction_loss-sample_lambda-1.0_res-256_classes-16_puncond-0.1_cfgw-1.0_seed-42",
    "generated_stage_folder": "stage5_pretrained_data",
}


PRESETS = {
    "baseline_intensity": {
        "use_acf": False,
        "max_real_samples_per_class": 50,
        "max_gen_samples_per_class": 0,
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "baseline_acf": {
        "use_acf": True,
        "max_real_samples_per_class": 50,
        "max_gen_samples_per_class": 0,
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "real25_intensity": {
        "use_acf": False,
        "max_real_samples_per_class": 25,
        "max_gen_samples_per_class": 0,
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "real75_intensity": {
        "use_acf": False,
        "max_real_samples_per_class": 75,
        "max_gen_samples_per_class": 0,
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "random_shift_16": {
        "use_acf": False,
        "max_real_samples_per_class": 50,
        "max_gen_samples_per_class": 0,
        "shift_amount": 16,
        "shift_mode": "random",
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "random_shift_32": {
        "use_acf": False,
        "max_real_samples_per_class": 50,
        "max_gen_samples_per_class": 0,
        "shift_amount": 32,
        "shift_mode": "random",
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "random_shift_48": {
        "use_acf": False,
        "max_real_samples_per_class": 50,
        "max_gen_samples_per_class": 0,
        "shift_amount": 48,
        "shift_mode": "random",
        "model_names": ("SimpleCNN", "ResNet18"),
    },
    "gen_aug_vv": {
        "use_acf": False,
        "max_real_samples_per_class": 25,
        "max_gen_samples_per_class": 50,
        "model_names": ("SimpleCNN", "ResNet18"),
        "save_confusion_matrices": True,
    },
}


PRESET_NAME = "baseline_intensity"


OVERRIDES = {
    # Example:
    # "random_seed": 100,
    # "real_data_root": "data/raw/dataset_z-5.00_sigma-1.1_l0-1.5_Nx-2048",
}


def build_config() -> dict:
    if PRESET_NAME not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown PRESET_NAME '{PRESET_NAME}'. Valid options: {valid}.")
    config = dict(BASE_CONFIG)
    config.update(PRESETS[PRESET_NAME])
    config.update(OVERRIDES)
    return config


def main() -> None:
    config_dict = build_config()
    config = ClassificationConfig(
        repo_root=REPO_ROOT,
        real_data_root=_resolve_repo_relative_path(REPO_ROOT, config_dict["real_data_root"]),
        real_metadata_filename=str(config_dict["real_metadata_filename"]),
        final_size=tuple(config_dict["final_size"]),
        canvas_size=tuple(config_dict["canvas_size"]),
        max_real_samples_per_class=int(config_dict["max_real_samples_per_class"]),
        max_gen_samples_per_class=int(config_dict["max_gen_samples_per_class"]),
        use_acf=bool(config_dict["use_acf"]),
        padding_mode=str(config_dict["padding_mode"]),
        shift_amount=int(config_dict["shift_amount"]),
        shift_mode=str(config_dict["shift_mode"]),
        eval_protocol=str(config_dict["eval_protocol"]),
        batch_size=int(config_dict["batch_size"]),
        num_epochs=int(config_dict["num_epochs"]),
        random_seed=int(config_dict["random_seed"]),
        save_confusion_matrices=bool(config_dict["save_confusion_matrices"]),
        model_names=tuple(config_dict["model_names"]),
        learning_rates=dict(config_dict["learning_rates"]),
        use_lr_finder=bool(config_dict["use_lr_finder"]),
        lr_finder_start_lr=float(config_dict["lr_finder_start_lr"]),
        lr_finder_end_lr=float(config_dict["lr_finder_end_lr"]),
        lr_finder_num_iter=int(config_dict["lr_finder_num_iter"]),
        generated_data_root=_resolve_repo_relative_path(REPO_ROOT, config_dict["generated_data_root"]),
        generated_stage_folder=str(config_dict["generated_stage_folder"]),
    )
    results = run_classification_experiment(config)
    print(f"Preset: {PRESET_NAME}")
    print(results)


if __name__ == "__main__":
    main()
