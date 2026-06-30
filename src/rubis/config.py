"""User-facing configuration objects for RUBIS."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .options import OutputOptions
from .rotation_profiles import solid


__all__ = [
    "CompositePolytropeConfig",
    "DeformationConfig",
    "LegacyModelConfig",
    "ModelConfig",
    "PolytropeConfig",
    "RotationConfig",
    "RotationProfile",
    "SolverMethod",
    "SolverOptions",
]


RotationProfile: TypeAlias = Callable[..., object]

SolverMethod: TypeAlias = Literal[
    "auto",
    "radial",
    "spheroidal",
]


@dataclass(kw_only=True)
class _PolytropeConfigBase(ABC):
    """Parameters shared by simple and composite polytropes."""

    radius: float = 1.0
    mass: float = 1.0
    n_points: int = 1001

    @property
    @abstractmethod
    def polytropic_indices(self) -> tuple[float, ...]:
        """Polytropic index of each region."""

    @property
    def n_regions(self) -> int:
        return len(self.polytropic_indices)

    @property
    def filename_stem(self) -> str:
        indices = "|".join(
            f"{index:.1f}"
            for index in self.polytropic_indices
        )
        return f"poly_|{indices}|"


@dataclass(kw_only=True)
class PolytropeConfig(_PolytropeConfigBase):
    """Configuration of a single polytrope."""

    index: float

    @property
    def polytropic_indices(self) -> tuple[float, ...]:
        return (self.index,)


@dataclass(kw_only=True)
class CompositePolytropeConfig(_PolytropeConfigBase):
    """Configuration of a composite polytrope."""

    indices: ArrayLike
    target_pressures: ArrayLike
    density_jumps: ArrayLike | None = None

    @property
    def polytropic_indices(self) -> tuple[float, ...]:
        if np.ndim(self.indices) == 0:
            return (float(self.indices),)

        return tuple(
            float(index)
            for index in self.indices
        )


@dataclass(kw_only=True)
class LegacyModelConfig:
    """Configuration of a model stored in the legacy RUBIS format."""

    filename: str
    directory: Path = Path("Models")

    @property
    def path(self) -> Path:
        return self.directory / self.filename

    @property
    def filename_stem(self) -> str:
        return Path(self.filename).stem


ModelConfig: TypeAlias = (
    PolytropeConfig
    | CompositePolytropeConfig
    | LegacyModelConfig
)


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
    """Complete configuration of a deformation calculation."""

    model: ModelConfig
    rotation: RotationConfig = field(
        default_factory=RotationConfig
    )
    solver: SolverOptions = field(
        default_factory=SolverOptions
    )
    output: OutputOptions = field(
        default_factory=OutputOptions
    )