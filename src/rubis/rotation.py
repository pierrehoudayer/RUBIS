from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Self

import numpy as np

from .config import RotationConfig


@dataclass(frozen=True, slots=True, kw_only=True)
class RotationState:
    """Bound rotation law with its current equatorial rate."""

    omega_eq: float
    _eval_phi_c: Callable
    _eval_omega: Callable

    def with_omega_eq(self, omega_eq: float) -> Self:
        """Return the same rotation law at a new equatorial rate."""
        return replace(
            self,
            omega_eq=omega_eq,
        )

    def phi_c(self, r, t):
        """Evaluate the centrifugal potential and its radial derivative."""
        return self._eval_phi_c(
            r,
            t,
            self.omega_eq,
        )
        
    def phi_c2d_with_derivative(self, r2d, t):
        """
        Evaluate the centrifugal potential and its radial derivative
        on a two-dimensional mapping.
        """
        return np.moveaxis(
            np.array([
                self.phi_c(r_j, t_j)
                for r_j, t_j in zip(r2d.T, t)
            ]),
            (0, 1, 2),
            (2, 0, 1),
        )

    def phi_c2d(self, r2d, t):
        """Evaluate the centrifugal potential on a two-dimensional mapping."""
        return self.phi_c2d_with_derivative(r2d, t)[0]

    def omega(self, r, t):
        """Evaluate the angular-velocity profile."""
        return self._eval_omega(
            r,
            t,
            self.omega_eq,
        )

    def omega2d(self, r2d, t):
        """Evaluate the angular velocity on a two-dimensional mapping."""
        return np.array([
            self.omega(r_j, t_j)
            for r_j, t_j in zip(r2d.T, t)
        ]).T
        
        
def _bind_rotation_profile(config: RotationConfig):
    """Bind the parameters required by a rotation profile."""
    profile = config.profile

    n_required = (
        profile.__code__.co_argcount
        - len(profile.__defaults__ or ())
    )
    n_parameters = n_required - 3

    parameters = (
        config.central_diff_rate,
        config.scale,
    )[:n_parameters]

    def eval_phi_c(r, t, omega_eq):
        return profile(
            r,
            t,
            omega_eq,
            *parameters,
        )

    def eval_omega(r, t, omega_eq):
        return profile(
            r,
            t,
            omega_eq,
            *parameters,
            return_profile=True,
        )

    return eval_phi_c, eval_omega


def initialize_rotation_state(
    config: RotationConfig,
    omega_eq: float = 0.0,
) -> RotationState:
    """Bind a rotation configuration into an evaluable state."""
    eval_phi_c, eval_omega = _bind_rotation_profile(config)

    return RotationState(
        omega_eq=omega_eq,
        _eval_phi_c=eval_phi_c,
        _eval_omega=eval_omega,
    )