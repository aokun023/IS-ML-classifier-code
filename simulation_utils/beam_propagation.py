"""Propagation and medium-generation utilities."""

from __future__ import annotations

import numpy as np


def ift2_centered(array: np.ndarray, dkx: float, dky: float, axes: tuple[int, int] = (-2, -1)) -> np.ndarray:
    """Compute the physically scaled inverse Fourier transform on centered data."""
    nx = array.shape[axes[0]]
    ny = array.shape[axes[1]]
    shifted = np.fft.ifftshift(array, axes=axes)
    transformed = np.fft.ifft2(shifted, axes=axes)
    centered = np.fft.fftshift(transformed, axes=axes)
    return centered * (dkx * dky * nx * ny) / (4.0 * np.pi**2)


def power_spectrum_ito_3d(kx: np.ndarray, ky: np.ndarray, xi: np.ndarray | float, l0: float = 1.0) -> np.ndarray:
    r"""Return the Gaussian power spectrum of the random medium."""
    radius_sq = np.abs(kx) ** 2 + np.abs(ky) ** 2 + np.abs(xi) ** 2
    return l0**2 * np.exp(-(l0**2) * radius_sq / (4.0 * np.pi**2))


def generate_random_medium_slice(
    power_spectrum_density: np.ndarray,
    dz: float,
    dkx: float,
    dky: float,
) -> np.ndarray:
    """Generate one real-valued slice of the random medium."""
    rand_fourier = (
        np.random.randn(*power_spectrum_density.shape)
        + 1j * np.random.randn(*power_spectrum_density.shape)
    )
    scaling = np.sqrt(
        dz * power_spectrum_density * (2.0 * np.pi / dkx) * (2.0 * np.pi / dky)
    )
    return np.real(ift2_centered(rand_fourier * scaling, dkx, dky))


def radial_average(image: np.ndarray) -> np.ndarray:
    """Compute the radial average of a centered square image."""
    image = np.asarray(image, dtype=np.float64)
    height, width = image.shape
    cy = height // 2
    cx = width // 2
    yy, xx = np.indices(image.shape)
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr_int = np.rint(rr).astype(np.int32)
    max_radius = min(height, width) // 2

    radial_sum = np.bincount(rr_int.ravel(), weights=image.ravel(), minlength=max_radius + 1)
    radial_count = np.bincount(rr_int.ravel(), minlength=max_radius + 1)
    profile = np.zeros(max_radius + 1, dtype=np.float64)
    valid = radial_count[: max_radius + 1] > 0
    profile[valid] = radial_sum[: max_radius + 1][valid] / radial_count[: max_radius + 1][valid]
    return profile


def estimate_correlation_length_2d(
    field: np.ndarray,
    dx: float,
    threshold: float = np.exp(-1.0),
) -> dict[str, np.ndarray | float]:
    """Estimate the correlation length from the periodic autocorrelation."""
    field = np.asarray(field, dtype=np.float64)
    centered = field - np.mean(field)
    acf2d = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(centered)) ** 2)).real

    center_value = acf2d[acf2d.shape[0] // 2, acf2d.shape[1] // 2]
    if center_value <= 0.0:
        return {
            "corr_length_pixels": 0.0,
            "corr_length_physical": 0.0,
            "threshold": float(threshold),
            "radial_acf": radial_average(np.zeros_like(acf2d)),
        }

    normalized_acf = acf2d / center_value
    radial_acf = radial_average(normalized_acf)
    below = np.where(radial_acf < threshold)[0]
    corr_pixels = int(below[0]) if len(below) > 0 else len(radial_acf) - 1
    return {
        "corr_length_pixels": float(corr_pixels),
        "corr_length_physical": float(corr_pixels * dx),
        "threshold": float(threshold),
        "radial_acf": radial_acf,
    }


def pwe_solver_splitting_ito_3d(
    dx: float,
    lx: float,
    dz: float,
    z_max: float,
    phi: np.ndarray,
    nu: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    """Propagate a field through a prescribed random medium."""
    x = np.arange(-lx / 2.0, lx / 2.0, dx)
    nx = len(x)
    dkx = 2.0 * np.pi / lx
    kx = dkx * np.arange(-nx / 2.0, nx / 2.0)
    kx_grid, ky_grid = np.meshgrid(kx, kx)

    z_steps = np.arange(0.0, z_max + dz, dz)
    nz = len(z_steps)

    propagator = np.exp(-1j * dz * (kx_grid**2 + ky_grid**2))
    propagator_fft = np.fft.fftshift(propagator)

    field = np.zeros((nx, nx, nz), dtype=np.complex128)
    field[:, :, 0] = phi
    for idx in range(nz - 1):
        screened = field[:, :, idx] * np.exp(1j * nu[:, :, idx])
        hat_screened = np.fft.fft2(screened)
        field[:, :, idx + 1] = np.fft.ifft2(propagator_fft * hat_screened)
        if verbose and idx % 10 == 0:
            print(f"Completed propagation step {idx}.")

    return field


def pwe_solver_splitting_ito_3d_efficient(
    dx: float,
    lx: float,
    dz: float,
    z_max: float,
    phi: np.ndarray,
    sigma: float,
    l0: float = 1.0,
    verbose: bool = False,
) -> np.ndarray:
    """Propagate a field while generating medium slices on the fly."""
    x = np.arange(-lx / 2.0, lx / 2.0, dx)
    nx = len(x)
    dkx = 2.0 * np.pi / lx
    kx = dkx * np.arange(-nx / 2.0, nx / 2.0)
    kx_grid, ky_grid = np.meshgrid(kx, kx)

    z_steps = np.arange(0.0, z_max + dz, dz)
    nz = len(z_steps)

    propagator = np.exp(-1j * dz * (kx_grid**2 + ky_grid**2))
    propagator_fft = np.fft.fftshift(propagator)
    power_spectrum_density = (
        sigma**2 * power_spectrum_ito_3d(kx_grid, ky_grid, 0.0, l0=l0) / 16.0
    )

    field = np.zeros((nx, nx, nz), dtype=np.complex128)
    field[:, :, 0] = phi
    for idx in range(nz - 1):
        medium_slice = generate_random_medium_slice(power_spectrum_density, dz, dkx, dkx)
        screened = field[:, :, idx] * np.exp(1j * medium_slice)
        hat_screened = np.fft.fft2(screened)
        field[:, :, idx + 1] = np.fft.ifft2(propagator_fft * hat_screened)
        if verbose and idx % 10 == 0:
            print(f"Completed propagation step {idx}.")

    return field[:, :, -1]
