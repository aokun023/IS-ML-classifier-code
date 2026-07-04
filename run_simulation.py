#!/usr/bin/env python3
"""Generate propagated-intensity data using the manuscript grid setting.

Warning:
    This script intentionally preserves the legacy multiprocessing behavior.
    On systems that start workers with ``fork``, workers inherit the same
    NumPy random state, so the generated files are not guaranteed to be
    independent turbulence realizations. Use this script only to reproduce
    the legacy multiprocessing experiment, not to generate a statistically
    independent dataset.
"""

from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_utils import BeamPropagationSimulation


def _format_float_tag(value: float) -> str:
    return f"{value:g}"


def _dataset_dirname(z_value: float, sigma_value: float, l0_value: float, nx_value: int) -> str:
    return (
        f"dataset_z-{z_value:.2f}_sigma-{_format_float_tag(sigma_value)}"
        f"_l0-{_format_float_tag(l0_value)}_Nx-{int(nx_value)}"
    )


def _downsample_to_256(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    height, width = image.shape
    target_size = 256

    if height == target_size and width == target_size:
        return image.copy()

    if height % target_size != 0 or width % target_size != 0:
        raise ValueError(
            "The image size must be divisible by 256 for average-pooling downsampling."
        )

    kernel_h = height // target_size
    kernel_w = width // target_size
    reshaped = image.reshape(target_size, kernel_h, target_size, kernel_w)
    return reshaped.mean(axis=(1, 3), dtype=np.float32)


def _generate_sample(task: dict) -> tuple[str, str, int]:
    """Generate one sample using the worker's inherited random state."""

    simulation = BeamPropagationSimulation(
        {
            "basis_modes": task["basis_modes"],
            "code": task["code"],
            "sim_params": task["sim_params"],
        }
    )
    simulation.run()
    _, final_intensity = simulation.get_results()
    final_small = _downsample_to_256(final_intensity)

    np.save(task["output_2048"], final_intensity.astype(np.float32))
    np.save(task["output_256"], final_small.astype(np.float32))
    return (
        task["relative_2048"],
        task["relative_256"],
        task["class_label"],
    )


CONFIG = {
    "basis_modes": [(0, 1), (1, 4), (0, -6), (1, 8)],
    "samples_per_class": 150,
    "num_processes": 4,
    "skip_existing": True,
    "sim_params": {
        "Lx": 64.0,
        "Nx": 2048,
        "z": 5.0,
        "dz": 1.0 / 32.0,
        "sigma": 1.1,
        "l0": 1.5,
        "w0": 4.0,
        "total_power": 1.0e5,
    },
}


def main() -> None:
    sim_params = dict(CONFIG["sim_params"])
    dataset_dir = REPO_ROOT / "data" / "raw" / _dataset_dirname(
        sim_params["z"],
        sim_params["sigma"],
        sim_params["l0"],
        sim_params["Nx"],
    )
    inputs_2048_dir = dataset_dir / "inputs_2048"
    inputs_256_dir = dataset_dir / "inputs_256"
    inputs_2048_dir.mkdir(parents=True, exist_ok=True)
    inputs_256_dir.mkdir(parents=True, exist_ok=True)

    rows_2048: list[tuple[str, int]] = []
    rows_256: list[tuple[str, int]] = []
    tasks: list[dict] = []
    num_basis = len(CONFIG["basis_modes"])
    num_classes = 2**num_basis - 1
    total_jobs = num_classes * int(CONFIG["samples_per_class"])

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
                continue

            tasks.append(
                {
                    "basis_modes": CONFIG["basis_modes"],
                    "code": code,
                    "sim_params": sim_params,
                    "class_label": class_label,
                    "output_2048": output_2048,
                    "output_256": output_256,
                    "relative_2048": relative_2048,
                    "relative_256": relative_256,
                }
            )

    with tqdm(
        total=total_jobs,
        initial=total_jobs - len(tasks),
        desc="Generating dataset",
    ) as progress:
        print(
            "WARNING: legacy reproduction mode preserves inherited worker RNG "
            "state; turbulence realizations may be duplicated."
        )
        with Pool(processes=int(CONFIG["num_processes"])) as pool:
            for relative_2048, relative_256, class_label in pool.imap_unordered(
                _generate_sample, tasks
            ):
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
        "num_processes": int(CONFIG["num_processes"]),
        "num_classes": num_classes,
        "sim_params": sim_params,
        "num_samples_2048": len(rows_2048),
        "num_samples_256": len(rows_256),
    }
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
