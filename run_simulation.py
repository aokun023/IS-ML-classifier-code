#!/usr/bin/env python3
"""Run one default beam-propagation simulation and save compact outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation_utils import BeamPropagationSimulation


CONFIG = {
    "basis_modes": [(0, 1), (1, 4), (0, -6), (1, 8)],
    "code": "1101",
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
    "output_dir": REPO_ROOT / "data" / "processed" / "default_simulation_1101",
}


def main() -> None:
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    simulation = BeamPropagationSimulation(
        {
            "basis_modes": CONFIG["basis_modes"],
            "code": CONFIG["code"],
            "sim_params": CONFIG["sim_params"],
        }
    )
    simulation.run()

    initial_intensity, final_intensity = simulation.get_results()
    medium_diag = simulation.estimate_medium_correlation(num_slices=8)

    np.save(output_dir / "initial_intensity.npy", initial_intensity)
    np.save(output_dir / "final_intensity.npy", final_intensity)
    np.save(output_dir / "final_field_real.npy", np.real(simulation.get_final_field()))
    np.save(output_dir / "final_field_imag.npy", np.imag(simulation.get_final_field()))

    summary = {
        "code": CONFIG["code"],
        "sim_params": dict(CONFIG["sim_params"]),
        "initial_intensity_shape": list(initial_intensity.shape),
        "final_intensity_shape": list(final_intensity.shape),
        "medium_corr_length_pixels_mean": medium_diag["mean_corr_length_pixels"],
        "medium_corr_length_pixels_std": medium_diag["std_corr_length_pixels"],
        "medium_corr_length_physical_mean": medium_diag["mean_corr_length_physical"],
        "medium_corr_length_physical_std": medium_diag["std_corr_length_physical"],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
