from dataclasses import dataclass, field


__all__ = [
    "DiagnosticOptions",
    "PlotOptions",
    "RadiativeFluxOptions",
    "ModelOutputOptions",
    "OutputOptions",
]


@dataclass(kw_only=True)
class DiagnosticOptions:
    """Optional numerical diagnostics."""

    virial_test: bool = False
    gravitational_moments: bool = False


@dataclass(kw_only=True)
class PlotOptions:
    """Model visualisation options."""

    show_harmonics: bool = False
    show_model: bool = False

    resolution: int = 501
    surfaces: bool = True
    field_cmap: str = "Stellar_r"
    surface_cmap: str = "Greys"


@dataclass(kw_only=True)
class RadiativeFluxOptions:
    """Radiative-flux computation and visualisation options."""

    enabled: bool = False
    plot_lines: bool = True
    origin: float = 0.05
    n_lines: int = 15
    show_effective_temperature: bool = True
    resolution: tuple[int, int] = (200, 100)
    cmap: str = "Stellar_r"


@dataclass(kw_only=True)
class ModelOutputOptions:
    """Model-file output options."""

    save: bool = False
    filename: str | None = None
    dimensional: bool = False


@dataclass(kw_only=True)
class OutputOptions:
    """Diagnostics, plots, and file output produced by a solver."""

    diagnostics: DiagnosticOptions = field(default_factory=DiagnosticOptions)
    plot: PlotOptions = field(default_factory=PlotOptions)
    flux: RadiativeFluxOptions = field(default_factory=RadiativeFluxOptions)
    model: ModelOutputOptions = field(default_factory=ModelOutputOptions)