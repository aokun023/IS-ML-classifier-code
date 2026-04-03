#!/usr/bin/env python3
"""Generate one propagated-intensity dataset for simulation, generation, and classification."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_utils import BeamPropagationSimulation


def _format_sigma(value: float) -> str:
    return f"{value:.0e}".replace("+0", "").replace("+", "")


def _dataset_dirname(z_value: float, sigma_value: float) -> str:
    return f"dataset_z-{z_value:.2f}_sigma-{_format_sigma(sigma_value)}"


def _downsample_to_256(image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    kernel_size = tensor.shape[-1] // 256
    if kernel_size > 1:
        tensor = torch.nn.AvgPool2d(kernel_size=kernel_size, stride=kernel_size)(tensor)
    return tensor.squeeze().numpy()


CONFIG = {
    "basis_modes": [(0, 1), (1, 4), (0, -6), (1, 8)],
    "samples_per_class": 150,
    "skip_existing": True,
    "sim_params": {
        "Lx": 64.0,
        "Nx": 2048,
        "z": 5.0,
        "dz": 1.0 / 32.0,
        "sigma": 5.0e-5,
        "l0": 1.5,
        "w0": 4.0,
        "total_power": 1.0,
    },
}


def main() -> None:
    sim_params = dict(CONFIG["sim_params"])
    dataset_dir = REPO_ROOT / "data" / "raw" / _dataset_dirname(sim_params["z"], sim_params["sigma"])
    inputs_2048_dir = dataset_dir / "inputs_2048"
    inputs_256_dir = dataset_dir / "inputs_256"
    inputs_2048_dir.mkdir(parents=True, exist_ok=True)
    inputs_256_dir.mkdir(parents=True, exist_ok=True)

    rows_2048: list[tuple[str, int]] = []
    rows_256: list[tuple[str, int]] = []
    num_basis = len(CONFIG["basis_modes"])
    num_classes = 2**num_basis - 1
    total_jobs = num_classes * int(CONFIG["samples_per_class"])

    with tqdm(total=total_jobs, desc="Generating dataset") as progress:
        for class_label in range(1, num_classes + 1):
            code = format(class_label, f"0{num_basis}b")
            for sample_idx in range(int(CONFIG["samples_per_class"])):
                filename = f"symbol_{code}_label_{class_label}_sample_{sample_idx:04d}.npy"
                output_2048 = inputs_2048_dir / filename
                output_256 = inputs_256_dir / filename
                relative_2048 = f"inputs_2048/{filename}"
                relative_256 = f"inputs_256/{filename}"

                if CONFIG["skip_existing"] and output_2048.exists() and output_256.exists():
                    rows_2048.append((relative_2048, class_label))
                    rows_256.append((relative_256, class_label))
                    progress.update(1)
                    continue

                simulation = BeamPropagationSimulation(
                    {
                        "basis_modes": CONFIG["basis_modes"],
                        "code": code,
                        "sim_params": sim_params,
                    }
                )
                simulation.run()
                _, final_intensity = simulation.get_results()
                final_small = _downsample_to_256(final_intensity)

                np.save(output_2048, final_intensity.astype(np.float32))
                np.save(output_256, final_small.astype(np.float32))
                rows_2048.append((relative_2048, class_label))
                rows_256.append((relative_256, class_label))
                progress.update(1)

    rows_2048.sort(key=lambda item: item[0])
    rows_256.sort(key=lambda item: item[0])
    for metadata_name, rows in (
        ("metadata_2048.csv", rows_2048),
        ("metadata_256.csv", rows_256),
        ("metadata.csv", rows_256),
    ):
        with open(dataset_dir / metadata_name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["input_path", "class_label"])
            writer.writerows(rows)

    summary = {
        "dataset_dir": dataset_dir.name,
        "basis_modes": list(CONFIG["basis_modes"]),
        "samples_per_class": int(CONFIG["samples_per_class"]),
        "num_classes": num_classes,
        "sim_params": sim_params,
        "num_samples_2048": len(rows_2048),
        "num_samples_256": len(rows_256),
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
