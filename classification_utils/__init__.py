"""Classification utilities for IS-ML experiments."""

from .datasets import (
    ACFTransform,
    NormalizeTensor,
    SpeckleGeneratedDataset,
    SpeckleRealDataset,
    compute_acf,
    compute_dataset_stats,
    load_generated_metadata,
)
from .experiment import ClassificationConfig, run_classification_experiment
from .models import MODEL_REGISTRY, ConfigurableResNet18, ConfigurableSimpleCNN
from .training import ClassificationTrainer, collect_predictions, count_parameters, set_seed

__all__ = [
    "ACFTransform",
    "ClassificationConfig",
    "ClassificationTrainer",
    "ConfigurableResNet18",
    "ConfigurableSimpleCNN",
    "MODEL_REGISTRY",
    "NormalizeTensor",
    "SpeckleGeneratedDataset",
    "SpeckleRealDataset",
    "collect_predictions",
    "compute_acf",
    "compute_dataset_stats",
    "count_parameters",
    "load_generated_metadata",
    "run_classification_experiment",
    "set_seed",
]
