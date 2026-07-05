"""HDF5 input for one-dimensional RUBIS models."""

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from numpy.typing import NDArray

from ..domains import DomainLayout, find_domains
from ..quadrature import integrate


__all__ = [
    "convert_legacy_model",
    "save_input_model",
]


FloatArray = NDArray[np.float64]

_FORMAT_NAME    = "RUBIS 1D model"
_FORMAT_VERSION = "1.0"
_UNIT_SYSTEM    = "cgs"

_R_UNIT        = "cm"
_RHO_UNIT      = "g cm-3"
_PRESSURE_UNIT = "dyn cm-2"


@dataclass(frozen=True, slots=True, kw_only=True)
class _InputModelData:
    """Validated dimensional data read from a native input file."""

    surface_pressure: float
    r: FloatArray
    rho: FloatArray
    additional_variables: tuple[FloatArray, ...]
    domains: DomainLayout
    mass: float


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


def _as_float_vector(
    name: str,
    values,
) -> FloatArray:
    """Return a finite one-dimensional float array."""
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")

    return values


def _validate_metadata(
    values: Iterable[str] | None,
    count: int,
    name: str,
) -> tuple[str | None, ...]:
    """Validate optional names or units for additional variables."""
    if values is None:
        return (None,) * count

    values = tuple(values)

    if len(values) != count:
        raise ValueError(
            f"{name} must contain one entry per additional variable."
        )

    if not all(isinstance(value, str) for value in values):
        raise TypeError(f"Every entry in {name} must be a string.")

    return values


def _validate_input_data(
    r,
    rho,
    surface_pressure: float,
    additional_variables=(),
) -> _InputModelData:
    """Validate dimensional model data and derive its mass."""
    r   = _as_float_vector("r", r)
    rho = _as_float_vector("rho", rho)

    if r.size != rho.size:
        raise ValueError("r and rho must have the same size.")

    if r.size == 0:
        raise ValueError("The input model must not be empty.")

    if np.any(r < 0.0):
        raise ValueError("The radial coordinate must be non-negative.")

    if np.any(rho < 0.0):
        raise ValueError("The density must be non-negative.")

    surface_pressure = float(surface_pressure)

    if not np.isfinite(surface_pressure):
        raise ValueError("surface_pressure must be finite.")

    if surface_pressure < 0.0:
        raise ValueError("surface_pressure must be non-negative.")

    domains = find_domains(r)

    if np.any(domains.domain_sizes < 4):
        raise ValueError(
            "Each radial domain must contain at least four points."
        )

    radius = r[-1]

    if radius <= 0.0:
        raise ValueError("The model radius must be positive.")

    variables = tuple(
        _as_float_vector(
            f"additional_variables[{index}]",
            values,
        )
        for index, values in enumerate(additional_variables)
    )

    if any(values.size != r.size for values in variables):
        raise ValueError(
            "Every additional variable must have the same size as r."
        )

    mass = 4 * np.pi * sum(
        integrate(
            x=r[domain],
            y=r[domain]**2 * rho[domain],
        )
        for domain in domains.domain_ranges
    )

    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(
            "The integrated model mass must be positive and finite."
        )

    return _InputModelData(
        surface_pressure=surface_pressure,
        r=r,
        rho=rho,
        additional_variables=variables,
        domains=domains,
        mass=float(mass),
    )


def _require_attribute(
    attrs: h5py.AttributeManager,
    name: str,
    expected: str,
):
    """Require a string attribute with a prescribed value."""
    value = _decode_string(attrs.get(name))

    if value != expected:
        raise ValueError(
            f"Invalid {name!r} attribute {value!r}; "
            f"expected {expected!r}."
        )


def _validate_file(
    file: h5py.File,
):
    """Validate the native RUBIS input format."""
    _require_attribute(
        file.attrs,
        "format_name",
        _FORMAT_NAME,
    )
    _require_attribute(
        file.attrs,
        "format_version",
        _FORMAT_VERSION,
    )
    _require_attribute(
        file.attrs,
        "unit_system",
        _UNIT_SYSTEM,
    )


def _read_additional_variables(
    group: h5py.Group,
) -> tuple[FloatArray, ...]:
    """Read additional material variables in their original order."""
    if "additional_variables" not in group:
        return ()

    additional = group["additional_variables"]
    count = int(additional.attrs.get("count", -1))

    if count < 0:
        raise ValueError(
            "The additional_variables group has no valid count."
        )

    expected = {str(index) for index in range(count)}

    if set(additional) != expected:
        raise ValueError(
            "The additional variables must be indexed consecutively "
            "from zero."
        )

    return tuple(
        additional[str(index)][...]
        for index in range(count)
    )


def _read_input_model(
    filename: str | Path,
) -> _InputModelData:
    """Read and validate a native dimensional input model."""
    path = Path(filename)

    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    with h5py.File(path, "r") as file:
        _validate_file(file)

        if "model" not in file:
            raise ValueError("The input file has no model group.")

        group = file["model"]

        if "surface_pressure" not in group.attrs:
            raise ValueError(
                "The model group has no surface_pressure attribute."
            )

        if "r" not in group or "rho" not in group:
            raise ValueError(
                "The model group must contain r and rho datasets."
            )

        _require_attribute(
            group.attrs,
            "surface_pressure_unit",
            _PRESSURE_UNIT,
        )
        _require_attribute(
            group["r"].attrs,
            "unit",
            _R_UNIT,
        )
        _require_attribute(
            group["rho"].attrs,
            "unit",
            _RHO_UNIT,
        )

        surface_pressure = group.attrs["surface_pressure"]
        r                = group["r"][...]
        rho              = group["rho"][...]
        variables        = _read_additional_variables(group)

    return _validate_input_data(
        r,
        rho,
        surface_pressure,
        variables,
    )


def save_input_model(
    filename: str | Path,
    *,
    r,
    rho,
    surface_pressure: float,
    additional_variables=(),
    names: Iterable[str] | None = None,
    units: Iterable[str] | None = None,
    overwrite: bool = False,
    compression: str | None = "gzip",
):
    """Save a dimensional one-dimensional model in native HDF5 format."""
    data = _validate_input_data(
        r,
        rho,
        surface_pressure,
        additional_variables,
    )
    names = _validate_metadata(
        names,
        len(data.additional_variables),
        "names",
    )
    units = _validate_metadata(
        units,
        len(data.additional_variables),
        "units",
    )

    mode = "w" if overwrite else "x"

    with h5py.File(filename, mode) as file:
        file.attrs["format_name"]    = _FORMAT_NAME
        file.attrs["format_version"] = _FORMAT_VERSION
        file.attrs["unit_system"]    = _UNIT_SYSTEM
        file.attrs["rubis_version"]  = _rubis_version()
        file.attrs["created_at"]     = datetime.now(
            timezone.utc
        ).isoformat()

        group = file.create_group("model")
        group.attrs["surface_pressure"] = data.surface_pressure
        group.attrs["surface_pressure_unit"] = _PRESSURE_UNIT

        options = _dataset_options(compression)

        r_dataset = group.create_dataset(
            "r",
            data=data.r,
            **options,
        )
        r_dataset.attrs["unit"] = _R_UNIT

        rho_dataset = group.create_dataset(
            "rho",
            data=data.rho,
            **options,
        )
        rho_dataset.attrs["unit"] = _RHO_UNIT

        additional = group.create_group("additional_variables")
        additional.attrs["count"] = len(
            data.additional_variables
        )

        for index, (values, variable_name, unit) in enumerate(
            zip(
                data.additional_variables,
                names,
                units,
            )
        ):
            dataset = additional.create_dataset(
                str(index),
                data=values,
                **options,
            )

            if variable_name is not None:
                dataset.attrs["name"] = variable_name

            if unit is not None:
                dataset.attrs["unit"] = unit


def convert_legacy_model(
    source: str | Path,
    destination: str | Path,
    *,
    names: Iterable[str] | None = None,
    units: Iterable[str] | None = None,
    overwrite: bool = False,
    compression: str | None = "gzip",
):
    """Convert a legacy RUBIS model to the native HDF5 format."""
    from ..config import LegacyModelConfig
    from ..initialization import G, initialize_model_1d

    source = Path(source)
    config = LegacyModelConfig(
        filename=source.name,
        directory=source.parent,
    )
    model = initialize_model_1d(config)

    radius_scale  = model.radius
    density_scale = model.mass / model.radius**3
    pressure_scale = (
        G * model.mass**2 / model.radius**4
    )

    save_input_model(
        destination,
        r=model.r * radius_scale,
        rho=model.rho * density_scale,
        surface_pressure=(
            model.surface_pressure * pressure_scale
        ),
        additional_variables=model.additional_variables,
        names=names,
        units=units,
        overwrite=overwrite,
        compression=compression,
    )
