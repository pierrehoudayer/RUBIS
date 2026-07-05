"""HDF5 input and output for RUBIS results."""

from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py

from ..config import SolverOptions
from ..diagnostics import (
    GravitationalMoments,
    ModelDiagnostics,
    VirialBalance,
)
from ..domains import find_domains
from ..models import Model2D, VacuumModel2D
from ..results import SolverInfo, SolverOutput


__all__ = [
    "load_diagnostics",
    "load_result",
    "load_solver_options",
    "save_result",
]


_FORMAT_NAME    = "RUBIS result"
_FORMAT_VERSION = "1.0"

_MODEL_SCALARS = (
    "G",
    "surface_pressure",
    "mass",
    "radius",
    "omega_eq",
)
_MODEL_ARRAYS = (
    "zeta",
    "t",
    "r2d",
    "rho",
    "p",
    "phi_eff",
    "phi_eff_z",
    "phi_g",
    "phi_g_z",
    "phi_c",
    "phi_c_z",
    "omega",
)

_VACUUM_SCALARS = (
    "G",
    "mass",
    "radius",
    "omega_eq",
)
_VACUUM_ARRAYS = (
    "zeta",
    "t",
    "r2d",
    "phi_g",
    "phi_g_z",
    "phi_c",
    "phi_c_z",
    "phi_eff",
    "phi_eff_z",
    "omega",
)

_INFO_SCALARS = (
    "method",
    "iterations",
    "tolerance",
    "error",
    "rotation_target",
    "elapsed_time",
)

_SOLVER_OPTION_FIELDS = (
    "method",
    "max_degree",
    "angular_resolution",
    "full_rate",
    "mapping_precision",
    "spline_order",
    "lagrange_order",
    "external_domain_res",
    "rescale_ab",
    "max_iterations",
    "verbose",
)

_VIRIAL_FIELDS = (
    "kinetic_energy",
    "potential_work",
    "thermodynamic_work",
    "surface_work",
)


def _rubis_version() -> str:
    """Return the installed RUBIS version."""
    try:
        return version("rubis")
    except PackageNotFoundError:
        return "unknown"


def _dataset_options(
    compression: str | None,
) -> dict[str, object]:
    """Return the common HDF5 dataset options."""
    if compression is None:
        return {}

    return {
        "compression": compression,
        "shuffle": True,
    }


def _decode_string(value):
    """Decode a string stored as bytes by HDF5."""
    if isinstance(value, bytes):
        return value.decode()

    return value


def _write_fields(
    group: h5py.Group,
    obj,
    scalars: tuple[str, ...],
    arrays: tuple[str, ...],
    *,
    compression: str | None,
):
    """Write scalar attributes and array datasets."""
    for name in scalars:
        group.attrs[name] = getattr(obj, name)

    options = _dataset_options(compression)

    for name in arrays:
        group.create_dataset(
            name,
            data=getattr(obj, name),
            **options,
        )


def _write_additional_variables(
    group: h5py.Group,
    variables: tuple,
    *,
    compression: str | None,
):
    """Write unnamed material variables in their original order."""
    additional = group.create_group("additional_variables")
    additional.attrs["count"] = len(variables)

    options = _dataset_options(compression)

    for index, values in enumerate(variables):
        additional.create_dataset(
            str(index),
            data=values,
            **options,
        )


def _read_additional_variables(
    group: h5py.Group,
) -> tuple:
    """Read unnamed material variables in their original order."""
    additional = group["additional_variables"]
    count = int(additional.attrs["count"])

    return tuple(
        additional[str(index)][...]
        for index in range(count)
    )


def _write_model(
    file: h5py.File,
    model: Model2D,
    *,
    compression: str | None,
):
    """Write the converged material model."""
    group = file.create_group("model")

    _write_fields(
        group,
        model,
        _MODEL_SCALARS,
        _MODEL_ARRAYS,
        compression=compression,
    )
    _write_additional_variables(
        group,
        model.additional_variables,
        compression=compression,
    )


def _read_model(
    file: h5py.File,
) -> Model2D:
    """Read the converged material model."""
    group = file["model"]

    scalars = {
        name: group.attrs[name]
        for name in _MODEL_SCALARS
    }
    arrays = {
        name: group[name][...]
        for name in _MODEL_ARRAYS
    }

    return Model2D(
        **scalars,
        **arrays,
        additional_variables=_read_additional_variables(group),
        domains=find_domains(arrays["zeta"]),
    )


def _write_vacuum(
    file: h5py.File,
    vacuum: VacuumModel2D | None,
    *,
    compression: str | None,
):
    """Write the exterior vacuum model when present."""
    if vacuum is None:
        return

    group = file.create_group("vacuum")

    _write_fields(
        group,
        vacuum,
        _VACUUM_SCALARS,
        _VACUUM_ARRAYS,
        compression=compression,
    )


def _read_vacuum(
    file: h5py.File,
) -> VacuumModel2D | None:
    """Read the exterior vacuum model when present."""
    if "vacuum" not in file:
        return None

    group = file["vacuum"]

    scalars = {
        name: group.attrs[name]
        for name in _VACUUM_SCALARS
    }
    arrays = {
        name: group[name][...]
        for name in _VACUUM_ARRAYS
    }

    return VacuumModel2D(
        **scalars,
        **arrays,
        domains=find_domains(arrays["zeta"]),
    )


def _write_solver_info(
    file: h5py.File,
    info: SolverInfo,
    *,
    compression: str | None,
):
    """Write convergence information."""
    group = file.create_group("solver")

    for name in _INFO_SCALARS:
        group.attrs[name] = getattr(info, name)

    group.create_dataset(
        "polar_radius_history",
        data=info.polar_radius_history,
        **_dataset_options(compression),
    )


def _read_solver_info(
    file: h5py.File,
) -> SolverInfo:
    """Read convergence information."""
    group = file["solver"]

    return SolverInfo(
        method=_decode_string(group.attrs["method"]),
        iterations=int(group.attrs["iterations"]),
        tolerance=float(group.attrs["tolerance"]),
        error=float(group.attrs["error"]),
        polar_radius_history=group["polar_radius_history"][...],
        rotation_target=float(group.attrs["rotation_target"]),
        elapsed_time=float(group.attrs["elapsed_time"]),
    )


def _write_solver_options(
    file: h5py.File,
    options: SolverOptions | None,
):
    """Write the requested solver options when available."""
    if options is None:
        return

    group = file["solver"].create_group("options")

    for name in _SOLVER_OPTION_FIELDS:
        group.attrs[name] = getattr(options, name)


def _read_solver_options(
    file: h5py.File,
) -> SolverOptions | None:
    """Read the requested solver options when available."""
    if "options" not in file["solver"]:
        return None

    attrs = file["solver/options"].attrs

    return SolverOptions(
        method=_decode_string(attrs["method"]),
        max_degree=int(attrs["max_degree"]),
        angular_resolution=int(attrs["angular_resolution"]),
        full_rate=int(attrs["full_rate"]),
        mapping_precision=float(attrs["mapping_precision"]),
        spline_order=int(attrs["spline_order"]),
        lagrange_order=int(attrs["lagrange_order"]),
        external_domain_res=int(attrs["external_domain_res"]),
        rescale_ab=bool(attrs["rescale_ab"]),
        max_iterations=int(attrs["max_iterations"]),
        verbose=bool(attrs["verbose"]),
    )


def _write_diagnostics(
    file: h5py.File,
    diagnostics: ModelDiagnostics | None,
    *,
    compression: str | None,
):
    """Write diagnostics that were explicitly provided."""
    if diagnostics is None:
        return

    virial = diagnostics.virial_balance
    moments = diagnostics.gravitational_moments

    if virial is None and moments is None:
        return

    group = file.create_group("diagnostics")

    if virial is not None:
        virial_group = group.create_group("virial_balance")

        for name in _VIRIAL_FIELDS:
            virial_group.attrs[name] = getattr(virial, name)

    if moments is not None:
        moments_group = group.create_group(
            "gravitational_moments"
        )
        options = _dataset_options(compression)

        moments_group.create_dataset(
            "degrees",
            data=moments.degrees,
            **options,
        )
        moments_group.create_dataset(
            "values",
            data=moments.values,
            **options,
        )


def _read_diagnostics(
    file: h5py.File,
) -> ModelDiagnostics | None:
    """Read stored diagnostics when available."""
    if "diagnostics" not in file:
        return None

    group = file["diagnostics"]
    virial = None
    moments = None

    if "virial_balance" in group:
        attrs = group["virial_balance"].attrs
        virial = VirialBalance(
            **{
                name: float(attrs[name])
                for name in _VIRIAL_FIELDS
            },
        )

    if "gravitational_moments" in group:
        moments_group = group["gravitational_moments"]
        moments = GravitationalMoments(
            degrees=moments_group["degrees"][...],
            values=moments_group["values"][...],
        )

    return ModelDiagnostics(
        virial_balance=virial,
        gravitational_moments=moments,
    )


def _validate_file(
    file: h5py.File,
):
    """Validate the RUBIS result format."""
    format_name = _decode_string(
        file.attrs.get("format_name")
    )
    format_version = _decode_string(
        file.attrs.get("format_version")
    )

    if format_name != _FORMAT_NAME:
        raise ValueError(
            "The file is not a RUBIS result."
        )

    if format_version != _FORMAT_VERSION:
        raise ValueError(
            "Unsupported RUBIS result format "
            f"{format_version!r}; expected {_FORMAT_VERSION!r}."
        )


def save_result(
    filename: str | Path,
    model: Model2D,
    vacuum: VacuumModel2D | None,
    info: SolverInfo,
    *,
    solver_options: SolverOptions | None = None,
    diagnostics: ModelDiagnostics | None = None,
    overwrite: bool = False,
    compression: str | None = "gzip",
):
    """Save a converged RUBIS result in HDF5 format."""
    mode = "w" if overwrite else "x"

    with h5py.File(filename, mode) as file:
        file.attrs["format_name"]    = _FORMAT_NAME
        file.attrs["format_version"] = _FORMAT_VERSION
        file.attrs["rubis_version"]  = _rubis_version()
        file.attrs["created_at"]     = datetime.now(
            timezone.utc
        ).isoformat()

        _write_model(
            file,
            model,
            compression=compression,
        )
        _write_vacuum(
            file,
            vacuum,
            compression=compression,
        )
        _write_solver_info(
            file,
            info,
            compression=compression,
        )
        _write_solver_options(
            file,
            solver_options,
        )
        _write_diagnostics(
            file,
            diagnostics,
            compression=compression,
        )


def load_result(
    filename: str | Path,
) -> SolverOutput:
    """Load a converged RUBIS result from HDF5 format."""
    with h5py.File(filename, "r") as file:
        _validate_file(file)

        model  = _read_model(file)
        vacuum = _read_vacuum(file)
        info   = _read_solver_info(file)

    return model, vacuum, info


def load_solver_options(
    filename: str | Path,
) -> SolverOptions | None:
    """Load stored solver options when available."""
    with h5py.File(filename, "r") as file:
        _validate_file(file)

        return _read_solver_options(file)


def load_diagnostics(
    filename: str | Path,
) -> ModelDiagnostics | None:
    """Load stored diagnostics when available."""
    with h5py.File(filename, "r") as file:
        _validate_file(file)

        return _read_diagnostics(file)
