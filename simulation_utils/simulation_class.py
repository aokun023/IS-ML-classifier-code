"""High-level beam-propagation simulation interface."""

from __future__ import annotations

import time

import numpy as np

from .beam_creation import generate_lg_beam_analytic
from .beam_propagation import (
    estimate_correlation_length_2d,
    generate_random_medium_slice,
    power_spectrum_ito_3d,
    pwe_solver_splitting_ito_3d,
)


class BeamPropagationSimulation:
    """End-to-end simulation of beam propagation through a random medium."""

    def __init__(self, config: dict, initial_phi_complex: np.ndarray | None = None):
        self.config = dict(config)
        self.params = self._resolve_grid_params(dict(config["sim_params"]))
        self.initial_phi_complex = initial_phi_complex

        self.nu: np.ndarray | None = None
        self.a_propagated: np.ndarray | None = None
        self.medium_diagnostics: dict | None = None

        self._setup_grids()

    @staticmethod
    def _resolve_grid_params(params: dict) -> dict:
        """Resolve a consistent grid from the supplied simulation parameters."""
        resolved = dict(params)
        if "Lx" not in resolved or "z" not in resolved:
            raise KeyError("Simulation parameters must include 'Lx' and 'z'.")

        lx = float(resolved["Lx"])
        z_max = float(resolved["z"])

        if "Nx" in resolved:
            nx = int(resolved["Nx"])
            if nx <= 0:
                raise ValueError("Nx must be positive.")
            dx = lx / nx
        elif "dx" in resolved:
            dx = float(resolved["dx"])
            nx = int(round(lx / dx))
            dx = lx / nx
        else:
            raise KeyError("Simulation parameters must include either 'Nx' or 'dx'.")

        if "Nz" in resolved:
            nz = int(resolved["Nz"])
            if nz < 2:
                raise ValueError("Nz must be at least 2.")
            dz = z_max / (nz - 1)
        elif "dz" in resolved:
            dz = float(resolved["dz"])
            nz = int(round(z_max / dz)) + 1
            dz = z_max / (nz - 1)
        else:
            raise KeyError("Simulation parameters must include either 'Nz' or 'dz'.")

        resolved["Lx"] = lx
        resolved["z"] = z_max
        resolved["Nx"] = nx
        resolved["Nz"] = nz
        resolved["dx"] = dx
        resolved["dz"] = dz
        return resolved

    def _setup_grids(self) -> None:
        """Set up the spatial and frequency grids."""
        self.dx = float(self.params["dx"])
        lx = float(self.params["Lx"])
        nx = int(self.params["Nx"])

        x = np.linspace(-lx / 2.0, lx / 2.0, num=nx, endpoint=False)
        self.xx, self.yy = np.meshgrid(x, x)

        freq = np.fft.fftshift(np.fft.fftfreq(nx, d=self.dx))
        self.fx, self.fy = np.meshgrid(freq, freq)
        self.kx = 2.0 * np.pi * self.fx
        self.ky = 2.0 * np.pi * self.fy

    def _create_source_beam(self) -> None:
        """Construct the superposed source beam from the prescribed code."""
        if self.initial_phi_complex is not None:
            return

        basis_modes = self.config["basis_modes"]
        code = self.config["code"]
        w0 = float(self.params["w0"])
        nx = int(self.params["Nx"])
        lx = float(self.params["Lx"])
        target_power = float(self.params.get("total_power", 1.0))

        basis_beams = [
            generate_lg_beam_analytic(p, ell, nx, lx, w0)
            for p, ell in basis_modes
        ]
        field = np.zeros((nx, nx), dtype=np.complex128)
        for idx, bit in enumerate(code):
            if bit == "1":
                field += basis_beams[idx]

        power = np.sum(np.abs(field) ** 2)
        if power > 0.0:
            field = field / np.sqrt(power)
        self.initial_phi_complex = field * np.sqrt(target_power)

    def _generate_medium(self) -> None:
        """Generate the three-dimensional random potential."""
        nx = int(self.params["Nx"])
        nz = int(self.params["Nz"])
        l0 = float(self.params.get("l0", 1.0))
        sigma = float(self.params["sigma"])
        dz = float(self.params["dz"])
        dkx = 2.0 * np.pi / float(self.params["Lx"])

        power_spectrum_density = sigma * power_spectrum_ito_3d(self.kx, self.ky, 0.0, l0=l0)
        self.nu = np.zeros((nx, nx, nz), dtype=np.float64)
        for idx in range(nz):
            self.nu[:, :, idx] = generate_random_medium_slice(power_spectrum_density, dz, dkx, dkx)

        self.medium_diagnostics = None

    def run(self, verbose: bool = False) -> None:
        """Run the full simulation pipeline."""
        self._create_source_beam()
        self._generate_medium()

        start_time = time.time()
        self.a_propagated = pwe_solver_splitting_ito_3d(
            dx=float(self.params["dx"]),
            lx=float(self.params["Lx"]),
            dz=float(self.params["dz"]),
            z_max=float(self.params["z"]),
            phi=self.initial_phi_complex,
            nu=self.nu,
            verbose=verbose,
        )
        elapsed = time.time() - start_time
        if verbose:
            print(f"Propagation finished in {elapsed:.2f} seconds.")

    def get_results(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the initial and final intensities."""
        if self.a_propagated is None or self.initial_phi_complex is None:
            raise RuntimeError("The simulation has not been run.")
        initial_intensity = np.abs(self.initial_phi_complex) ** 2
        final_intensity = np.abs(self.a_propagated[:, :, -1]) ** 2
        return initial_intensity, final_intensity

    def get_final_field(self) -> np.ndarray:
        """Return the final propagated complex field."""
        if self.a_propagated is None:
            raise RuntimeError("The simulation has not been run.")
        return self.a_propagated[:, :, -1]

    def get_initial_spectrum(self) -> np.ndarray:
        """Return the power spectrum of the source beam."""
        if self.initial_phi_complex is None:
            raise RuntimeError("The source beam has not been initialized.")
        field_k = np.fft.fftshift(np.fft.fft2(self.initial_phi_complex, norm="ortho"))
        return np.abs(field_k) ** 2

    def get_final_spectrum(self) -> np.ndarray:
        """Return the power spectrum of the final propagated field."""
        if self.a_propagated is None:
            raise RuntimeError("The simulation has not been run.")
        field_k = np.fft.fftshift(np.fft.fft2(self.get_final_field(), norm="ortho"))
        return np.abs(field_k) ** 2

    def get_medium_slice(self, slice_index: int | None = None) -> np.ndarray:
        """Return one slice of the random medium."""
        if self.nu is None:
            raise RuntimeError("The random medium has not been generated.")
        if slice_index is None:
            slice_index = self.nu.shape[2] // 2
        return self.nu[:, :, int(slice_index)]

    def estimate_medium_correlation(self, num_slices: int = 8) -> dict:
        """Estimate the medium correlation length from several slices."""
        if self.nu is None:
            raise RuntimeError("The random medium has not been generated.")

        nz = self.nu.shape[2]
        sample_count = max(1, min(int(num_slices), nz))
        slice_indices = np.linspace(0, nz - 1, num=sample_count, dtype=int)

        slice_diagnostics = []
        for idx in slice_indices:
            diag = estimate_correlation_length_2d(self.nu[:, :, idx], float(self.params["dx"]))
            diag["slice_index"] = int(idx)
            slice_diagnostics.append(diag)

        corr_pixels = np.array([item["corr_length_pixels"] for item in slice_diagnostics], dtype=np.float64)
        corr_physical = np.array([item["corr_length_physical"] for item in slice_diagnostics], dtype=np.float64)

        summary = {
            "input_l0_physical": float(self.params.get("l0", 1.0)),
            "input_l0_pixels": float(self.params.get("l0", 1.0) / float(self.params["dx"])),
            "mean_corr_length_pixels": float(np.mean(corr_pixels)),
            "std_corr_length_pixels": float(np.std(corr_pixels)),
            "mean_corr_length_physical": float(np.mean(corr_physical)),
            "std_corr_length_physical": float(np.std(corr_physical)),
            "num_slices_used": int(sample_count),
            "slice_diagnostics": slice_diagnostics,
        }
        self.medium_diagnostics = summary
        return summary
