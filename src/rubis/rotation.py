"""Rotation-state initialization and evaluation."""

from dataclasses import dataclass, replace
from functools import partial
from typing import Self

import numpy as np

from .config import RotationConfig, RotationProfile


__all__ = [
    "RotationState",
    "initialize_rotation_state",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class RotationState:
    """Bound rotation law with its current equatorial rate."""

    omega_eq: float
    _profile: RotationProfile

    def with_omega_eq(self, omega_eq: float) -> Self:
        """Return the same rotation law at a new equatorial rate."""
        return replace(self, omega_eq=omega_eq)

    def phi_c(self, r, t):
        """Evaluate the centrifugal potential and its radial derivative."""
        return self._profile(r, t, self.omega_eq)

    def phi_c2d_with_derivative(self, r2d, t):
        """Evaluate the centrifugal potential and its radial derivative."""
        return self.phi_c(r2d, t)

    def phi_c2d(self, r2d, t):
        """Evaluate the centrifugal potential on a two-dimensional mapping."""
        phi_c, _ = self.phi_c2d_with_derivative(r2d, t)

        return phi_c

    def omega(self, r, t):
        """Evaluate the angular-velocity profile."""
        return self._profile(
            r,
            t,
            self.omega_eq,
            return_profile=True,
        )

    def omega2d(self, r2d, t):
        """Evaluate the angular velocity on a two-dimensional mapping."""
        return self.omega(r2d, t)


def initialize_rotation_state(
    config: RotationConfig,
    omega_eq: float = 0.0,
) -> RotationState:
    """Bind a rotation configuration into an evaluable state."""
    profile = partial(
        config.profile,
        **config.profile_parameters,
    )

    return RotationState(
        omega_eq=omega_eq,
        _profile=profile,
    )