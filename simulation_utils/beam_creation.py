"""Beam-construction utilities."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import genlaguerre


DEFAULT_WAVELENGTH = 4.0 * np.pi


def beam_radius_at_distance(w0: float, z: float, wavelength: float = DEFAULT_WAVELENGTH) -> float:
    """Return the Gaussian-beam radius at axial distance ``z``."""
    rayleigh_range = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1.0 + (z / rayleigh_range) ** 2)


def generate_lg_beam_analytic(
    p: int,
    ell: int,
    grid_size: int,
    grid_length: float,
    w0: float,
    distance: float = 0.0,
    normalize: bool = True,
    wavelength: float = DEFAULT_WAVELENGTH,
) -> np.ndarray:
    """Generate a Laguerre-Gaussian beam on a square grid.

    Parameters
    ----------
    p, ell:
        Radial and azimuthal mode indices.
    grid_size, grid_length:
        Number of grid points per side and physical side length.
    w0:
        Beam waist at ``z = 0``.
    distance:
        Propagation distance used in the analytic beam expression.
    normalize:
        If ``True``, normalize the discrete beam power to one.
    wavelength:
        Wavelength in the non-dimensionalized model.
    """
    dx = grid_length / grid_size
    x = np.arange(-grid_length / 2.0, grid_length / 2.0, dx)
    xx, yy = np.meshgrid(x, x)

    rho = np.sqrt(xx**2 + yy**2)
    theta = np.arctan2(yy, xx)

    current_w = beam_radius_at_distance(w0, z=distance, wavelength=wavelength)
    rayleigh_range = np.pi * w0**2 / wavelength
    gouy_phase = np.arctan2(distance, rayleigh_range)

    norm_const = np.sqrt(
        (2.0 * math.factorial(p)) / (np.pi * math.factorial(p + abs(ell)))
    )
    radial_term = (np.sqrt(2.0) * rho / current_w) ** abs(ell)
    laguerre_term = genlaguerre(p, abs(ell))(2.0 * rho**2 / current_w**2)
    gaussian_term = np.exp(-rho**2 / current_w**2)

    if distance != 0.0:
        curvature_radius = distance * (1.0 + (rayleigh_range / distance) ** 2)
        curvature_phase = (2.0 * np.pi / wavelength) * rho**2 / (2.0 * curvature_radius)
    else:
        curvature_phase = 0.0

    phase = ell * theta - gouy_phase * (2 * p + abs(ell) + 1) + curvature_phase
    beam = (
        norm_const
        / current_w
        * radial_term
        * laguerre_term
        * gaussian_term
        * np.exp(1j * phase)
    )

    if normalize:
        power = np.sum(np.abs(beam) ** 2) * dx**2
        if power > 0.0:
            beam = beam / np.sqrt(power)

    return beam
