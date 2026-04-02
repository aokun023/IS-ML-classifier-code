"""Minimal regression tests for the simulation module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_utils import (  # noqa: E402
    BeamPropagationSimulation,
    estimate_correlation_length_2d,
    generate_lg_beam_analytic,
    power_spectrum_ito_3d,
)


class SimulationModuleTest(unittest.TestCase):
    """Fast interface-level checks for the migrated simulation code."""

    def test_generate_lg_beam_has_unit_discrete_power(self) -> None:
        beam = generate_lg_beam_analytic(
            p=0,
            ell=1,
            grid_size=64,
            grid_length=16.0,
            w0=2.0,
        )
        dx = 16.0 / 64.0
        power = np.sum(np.abs(beam) ** 2) * dx**2
        self.assertTrue(np.isclose(power, 1.0, rtol=5e-3, atol=5e-3))

    def test_power_spectrum_is_nonnegative(self) -> None:
        k = np.linspace(-1.0, 1.0, 8)
        kx, ky = np.meshgrid(k, k)
        psd = power_spectrum_ito_3d(kx, ky, 0.0, l0=1.5)
        self.assertEqual(psd.shape, kx.shape)
        self.assertTrue(np.all(psd >= 0.0))

    def test_small_simulation_runs_and_returns_expected_shapes(self) -> None:
        config = {
            "basis_modes": [(0, 1), (1, 4), (0, -6), (1, 8)],
            "code": "1101",
            "sim_params": {
                "Lx": 8.0,
                "Nx": 32,
                "z": 0.25,
                "Nz": 5,
                "sigma": 0.0,
                "l0": 1.5,
                "w0": 1.5,
                "total_power": 1.0,
            },
        }
        simulation = BeamPropagationSimulation(config)
        simulation.run()

        initial_intensity, final_intensity = simulation.get_results()
        self.assertEqual(initial_intensity.shape, (32, 32))
        self.assertEqual(final_intensity.shape, (32, 32))
        self.assertEqual(simulation.get_final_field().shape, (32, 32))
        self.assertEqual(simulation.get_medium_slice().shape, (32, 32))

    def test_correlation_length_estimator_returns_required_keys(self) -> None:
        field = np.ones((16, 16), dtype=np.float64)
        diagnostics = estimate_correlation_length_2d(field, dx=0.5)
        self.assertIn("corr_length_pixels", diagnostics)
        self.assertIn("corr_length_physical", diagnostics)
        self.assertIn("radial_acf", diagnostics)


if __name__ == "__main__":
    unittest.main()
