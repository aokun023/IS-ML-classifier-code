#!/usr/bin/env python3
"""Run the default IS-ML classification experiment."""

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


CONFIG = {
    "real_data_root": "data/raw/dataset_z-5.00_sigma-5e-05",
    "real_metadata_filename": "metadata_256.csv",
    "final_size": (64, 64),
    "canvas_size": (256, 256),
    "max_real_samples_per_class": 25,
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
    "generated_data_root": "results/generation_z-5.00_sigma-5e-05_pred-v_prediction_loss-sample_lambda-1.0_res-256_seed-42",
    "generated_stage_folder": "stage5_pretrained_data",
}


def main() -> None:
    config = ClassificationConfig(
        repo_root=REPO_ROOT,
        real_data_root=_resolve_repo_relative_path(REPO_ROOT, CONFIG["real_data_root"]),
        real_metadata_filename=str(CONFIG["real_metadata_filename"]),
        final_size=tuple(CONFIG["final_size"]),
        canvas_size=tuple(CONFIG["canvas_size"]),
        max_real_samples_per_class=int(CONFIG["max_real_samples_per_class"]),
        max_gen_samples_per_class=int(CONFIG["max_gen_samples_per_class"]),
        use_acf=bool(CONFIG["use_acf"]),
        padding_mode=str(CONFIG["padding_mode"]),
        shift_amount=int(CONFIG["shift_amount"]),
        shift_mode=str(CONFIG["shift_mode"]),
        eval_protocol=str(CONFIG["eval_protocol"]),
        batch_size=int(CONFIG["batch_size"]),
        num_epochs=int(CONFIG["num_epochs"]),
        random_seed=int(CONFIG["random_seed"]),
        save_confusion_matrices=bool(CONFIG["save_confusion_matrices"]),
        model_names=tuple(CONFIG["model_names"]),
        learning_rates=dict(CONFIG["learning_rates"]),
        generated_data_root=_resolve_repo_relative_path(REPO_ROOT, CONFIG["generated_data_root"]),
        generated_stage_folder=str(CONFIG["generated_stage_folder"]),
    )
    results = run_classification_experiment(config)
    print(results)


if __name__ == "__main__":
    main()
