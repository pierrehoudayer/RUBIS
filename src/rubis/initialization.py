"""Initialisation of one-dimensional models for RUBIS solvers."""

from pathlib import Path

import numpy as np

from .config import (
    CompositePolytropeConfig,
    LegacyModelConfig,
    ModelConfig,
    PolytropeConfig,
)
from .domains import find_domains
from .models import Model1D
from .quadrature import integrate
from .polytrope import build_polytrope


__all__ = [
    "initialize_model_1d",
]


G = 6.67384e-8


def _initialize_polytropic_model(
    config: PolytropeConfig | CompositePolytropeConfig,
) -> Model1D:
    source = build_polytrope(config)

    mass = config.mass
    radius = config.radius

    r = source.r / radius
    rho = source.rho / (mass / radius**3)

    return Model1D(
        G=G,
        surface_pressure=source.p[-1] / (
            G * mass**2 / radius**4
        ),
        mass=mass,
        radius=radius,
        r=r,
        rho=rho,
        domains=find_domains(r),
    )
    
    
def _initialize_legacy_model(
    config: LegacyModelConfig,
) -> Model1D:
    path = config.path

    if not path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {path}"
        )

    surface_pressure, declared_points = np.genfromtxt(
        path,
        max_rows=2,
        unpack=True,
    )

    r, rho, *additional_variables = np.genfromtxt(
        path,
        skip_header=2,
        unpack=True,
    )

    declared_points = int(declared_points)

    if r.size != declared_points:
        raise ValueError(
            f"{path} declares {declared_points} radial points "
            f"but contains {r.size}."
        )

    domains = find_domains(r)

    radius = r[-1]
    if radius <= 0:
        raise ValueError(
            "The model radius must be positive."
        )

    mass = 4 * np.pi * sum(
        integrate(
            x=r[D],
            y=r[D]**2 * rho[D],
        )
        for D in domains.domain_ranges
    )

    if mass <= 0:
        raise ValueError(
            "The integrated model mass must be positive."
        )

    r = r / radius
    rho = rho / (mass / radius**3)

    surface_pressure /= mass**2 / radius**4

    # Preserve the historical RUBIS file convention.
    if not np.allclose(radius, 1.0):
        surface_pressure /= G

    return Model1D(
        G=G,
        surface_pressure=surface_pressure,
        mass=mass,
        radius=radius,
        r=r,
        rho=rho,
        domains=find_domains(r),
        additional_variables=tuple(additional_variables),
    )
    
    
def initialize_model_1d(
    config: ModelConfig,
) -> Model1D:
    """Construct, read, and normalise a one-dimensional model."""
    if isinstance(
        config,
        (PolytropeConfig, CompositePolytropeConfig),
    ):
        return _initialize_polytropic_model(config)

    if isinstance(config, LegacyModelConfig):
        return _initialize_legacy_model(config)

    raise TypeError(
        "config must be a PolytropeConfig, "
        "CompositePolytropeConfig or LegacyModelConfig."
    )