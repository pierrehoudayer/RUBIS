from pathlib import Path

import h5py
import numpy as np
import pytest

from rubis.config import HDF5ModelConfig, LegacyModelConfig
from rubis.initialization import G, initialize_model_1d
from rubis.io import convert_legacy_model, save_input_model


def make_input_data():
    r = np.linspace(0.0, 2.0, 9)
    rho = np.array([
        4.0,
        3.4,
        2.9,
        3.1,
        2.2,
        1.6,
        1.1,
        0.6,
        0.2,
    ])
    temperature = np.linspace(1.0e7, 1.0e4, r.size)

    return r, rho, temperature


def test_hdf5_model_config_builds_path():
    config = HDF5ModelConfig(path="models/sun.h5")

    assert config.path == Path("models/sun.h5")
    assert config.filename_stem == "sun"


def test_save_input_model_initializes_normalised_model(tmp_path):
    path = tmp_path / "model.h5"
    r, rho, temperature = make_input_data()
    surface_pressure = 2.5e12

    save_input_model(
        path,
        r=r,
        rho=rho,
        surface_pressure=surface_pressure,
        additional_variables=(temperature,),
        names=("temperature",),
        units=("K",),
    )

    model = initialize_model_1d(
        HDF5ModelConfig(path=path)
    )

    radius = r[-1]
    density_scale = model.mass / radius**3
    pressure_scale = G * model.mass**2 / radius**4

    np.testing.assert_allclose(
        model.r,
        r / radius,
    )
    np.testing.assert_allclose(
        model.rho,
        rho / density_scale,
    )
    np.testing.assert_allclose(
        model.surface_pressure,
        surface_pressure / pressure_scale,
    )
    np.testing.assert_array_equal(
        model.additional_variables[0],
        temperature,
    )

    assert model.radius == radius
    assert model.n_domains == 1


def test_input_file_contains_explicit_units_and_metadata(tmp_path):
    path = tmp_path / "model.h5"
    r, rho, temperature = make_input_data()

    save_input_model(
        path,
        r=r,
        rho=rho,
        surface_pressure=0.0,
        additional_variables=(temperature,),
        names=("temperature",),
        units=("K",),
    )

    with h5py.File(path, "r") as file:
        assert file.attrs["format_name"] == "RUBIS 1D model"
        assert file.attrs["format_version"] == "1.0"
        assert file.attrs["unit_system"] == "cgs"
        assert isinstance(file.attrs["rubis_version"], str)
        assert isinstance(file.attrs["created_at"], str)

        group = file["model"]
        additional = group["additional_variables"]

        assert group.attrs["surface_pressure_unit"] == "dyn cm-2"
        assert group["r"].attrs["unit"] == "cm"
        assert group["rho"].attrs["unit"] == "g cm-3"
        assert additional.attrs["count"] == 1
        assert additional["0"].attrs["name"] == "temperature"
        assert additional["0"].attrs["unit"] == "K"


def test_input_model_accepts_nonmonotonic_density(tmp_path):
    path = tmp_path / "model.h5"
    r, rho, _ = make_input_data()

    assert np.any(np.diff(rho) > 0.0)

    save_input_model(
        path,
        r=r,
        rho=rho,
        surface_pressure=0.0,
    )

    model = initialize_model_1d(
        HDF5ModelConfig(path=path)
    )

    assert model.n_points == r.size


def test_input_model_preserves_discontinuous_domains(tmp_path):
    path = tmp_path / "model.h5"
    r = np.array([
        0.0,
        0.15,
        0.30,
        0.45,
        0.60,
        0.60,
        0.70,
        0.80,
        0.90,
        1.00,
    ])
    rho = np.array([
        5.0,
        4.8,
        4.4,
        4.0,
        3.6,
        2.0,
        1.8,
        1.5,
        1.1,
        0.7,
    ])

    save_input_model(
        path,
        r=r,
        rho=rho,
        surface_pressure=0.0,
    )

    model = initialize_model_1d(
        HDF5ModelConfig(path=path)
    )

    assert model.n_domains == 2
    assert model.domains.has_interfaces
    np.testing.assert_array_equal(
        model.domains.domain_sizes,
        (5, 5),
    )


@pytest.mark.parametrize(
    "radius, surface_pressure",
    [
        (1.0, 0.25),
        (2.0, 2.5e12),
    ],
)
def test_convert_legacy_model_preserves_initialisation(
    tmp_path,
    radius,
    surface_pressure,
):
    source = tmp_path / "legacy.txt"
    destination = tmp_path / "model.h5"

    r = np.linspace(0.0, radius, 9)
    rho = np.linspace(3.0, 0.5, r.size)
    temperature = np.linspace(100.0, 900.0, r.size)

    with source.open("w") as file:
        file.write(f"{surface_pressure}\n{r.size}\n")
        np.savetxt(
            file,
            np.column_stack((
                r,
                rho,
                temperature,
            )),
        )

    legacy = initialize_model_1d(
        LegacyModelConfig(
            filename=source.name,
            directory=source.parent,
        )
    )

    convert_legacy_model(
        source,
        destination,
        names=("temperature",),
        units=("K",),
    )

    native = initialize_model_1d(
        HDF5ModelConfig(path=destination)
    )

    assert native.G == legacy.G
    np.testing.assert_allclose(native.mass, legacy.mass)
    np.testing.assert_allclose(native.radius, legacy.radius)
    np.testing.assert_allclose(
        native.surface_pressure,
        legacy.surface_pressure,
    )
    np.testing.assert_allclose(native.r, legacy.r)
    np.testing.assert_allclose(native.rho, legacy.rho)
    np.testing.assert_array_equal(
        native.additional_variables[0],
        legacy.additional_variables[0],
    )


def test_save_input_model_does_not_overwrite_by_default(tmp_path):
    path = tmp_path / "model.h5"
    r, rho, _ = make_input_data()

    save_input_model(
        path,
        r=r,
        rho=rho,
        surface_pressure=0.0,
    )

    with pytest.raises(FileExistsError):
        save_input_model(
            path,
            r=r,
            rho=rho,
            surface_pressure=0.0,
        )


def test_save_input_model_rejects_decreasing_radius(tmp_path):
    r, rho, _ = make_input_data()
    r[4] = r[3] - 0.1

    with pytest.raises(ValueError, match="non-decreasing"):
        save_input_model(
            tmp_path / "model.h5",
            r=r,
            rho=rho,
            surface_pressure=0.0,
        )


def test_save_input_model_rejects_negative_density(tmp_path):
    r, rho, _ = make_input_data()
    rho[3] = -1.0

    with pytest.raises(ValueError, match="non-negative"):
        save_input_model(
            tmp_path / "model.h5",
            r=r,
            rho=rho,
            surface_pressure=0.0,
        )


def test_save_input_model_rejects_short_domains(tmp_path):
    r = np.array([
        0.0,
        0.5,
        0.5,
        0.75,
        1.0,
    ])
    rho = np.ones_like(r)

    with pytest.raises(ValueError, match="at least four points"):
        save_input_model(
            tmp_path / "model.h5",
            r=r,
            rho=rho,
            surface_pressure=0.0,
        )


def test_hdf5_reader_rejects_inconsistent_units(tmp_path):
    path = tmp_path / "model.h5"
    r, rho, _ = make_input_data()

    save_input_model(
        path,
        r=r,
        rho=rho,
        surface_pressure=0.0,
    )

    with h5py.File(path, "r+") as file:
        file["model/r"].attrs["unit"] = "m"

    with pytest.raises(ValueError, match="expected 'cm'"):
        initialize_model_1d(
            HDF5ModelConfig(path=path)
        )
