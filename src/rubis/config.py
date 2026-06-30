"""Configuration of a centrifugal-deformation calculation."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .models import PolytropicModelConfig
from .options import OutputOptions
from .rotation_profiles import solid


__all__ = [
    "DeformationConfig",
    "ModelInput",
    "RotationConfig",
    "RotationProfile",
    "SolverMethod",
    "SolverOptions",
]


ModelInput: TypeAlias = PolytropicModelConfig | str
RotationProfile: TypeAlias = Callable[..., object]
SolverMethod: TypeAlias = Literal[
    "auto",
    "radial",
    "spheroidal",
]


@dataclass(kw_only=True)
class RotationConfig:
    """Rotation profile and target rate."""

    profile: RotationProfile = solid
    target: float = 0.0
    central_diff_rate: float = 0.0
    scale: float = 1.0


@dataclass(kw_only=True)
class SolverOptions:
    """Numerical options controlling the deformation solver."""

    method: SolverMethod = "auto"

    max_degree: int = 101
    angular_resolution: int = 101
    full_rate: int = 3
    mapping_precision: float = 1.0e-10

    spline_order: int = 5
    lagrange_order: int = 3

    external_domain_res: int = 201
    rescale_ab: bool = True
    max_iterations: int = 200


@dataclass(kw_only=True)
class DeformationConfig:
    """Complete input configuration for a deformation calculation."""

    model: ModelInput
    rotation: RotationConfig = field(default_factory=RotationConfig)
    solver: SolverOptions = field(default_factory=SolverOptions)
    output: OutputOptions = field(default_factory=OutputOptions)