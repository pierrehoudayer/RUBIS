import h5py
import numpy as np
import pytest

from rubis.config import SolverOptions
from rubis.diagnostics import (
    GravitationalMoments,
    ModelDiagnostics,
    VirialBalance,
)
from rubis.domains import find_domains
from rubis.io import (
    load_diagnostics,
    load_result,
    load_solver_options,
    save_result,
)
from rubis.models import Model2D, VacuumModel2D
from rubis.results import SolverInfo


def make_model():
    zeta = np.array([
        0.0,
        0.4,
        0.4,
        0.7,
        1.0,
    ])
    t = np.array([
        -0.5,
        0.0,
        0.5,
    ])
    r2d = zeta[:, None] * (
        1.0 - 0.1*(1 - t**2)
    )
    field_2d = np.arange(
        r2d.size,
        dtype=float,
    ).reshape(r2d.shape)

    return Model2D(
        G=6.67384e-8,
        surface_pressure=1.0e-6,
        mass=2.0,
        radius=3.0,
        omega_eq=0.4,
        zeta=zeta,
        t=t,
        r2d=r2d,
        rho=np.linspace(2.0, 0.5, zeta.size),
        p=np.linspace(3.0, 1.0, zeta.size),
        additional_variables=(
            np.linspace(10.0, 20.0, zeta.size),
        ),
        phi_eff=np.linspace(-2.0, -1.0, zeta.size),
        phi_eff_z=np.linspace(1.0, 2.0, zeta.size),
        phi_g=field_2d,
        phi_g_z=field_2d + 1.0,
        phi_c=field_2d + 2.0,
        phi_c_z=field_2d + 3.0,
        omega=field_2d + 4.0,
        domains=find_domains(zeta),
    )


def make_vacuum():
    zeta = np.linspace(1.0, 2.0, 4)
    t = np.array([
        -0.5,
        0.0,
        0.5,
    ])
    r2d = zeta[:, None] * np.ones_like(t)
    field_2d = np.arange(
        r2d.size,
        dtype=float,
    ).reshape(r2d.shape)

    return VacuumModel2D(
        G=6.67384e-8,
        mass=2.0,
        radius=3.0,
        omega_eq=0.4,
        zeta=zeta,
        t=t,
        r2d=r2d,
        phi_g=field_2d,
        phi_g_z=field_2d + 1.0,
        phi_c=field_2d + 2.0,
        phi_c_z=field_2d + 3.0,
        phi_eff=field_2d + 4.0,
        phi_eff_z=field_2d + 5.0,
        omega=field_2d + 6.0,
        domains=find_domains(zeta),
    )


def make_info():
    return SolverInfo(
        method="spheroidal",
        iterations=7,
        tolerance=1.0e-10,
        error=2.0e-11,
        polar_radius_history=np.array([
            1.0,
            0.9,
            0.89,
        ]),
        rotation_target=0.4,
        elapsed_time=1.2,
    )


def assert_domains_equal(actual, expected):
    np.testing.assert_array_equal(
        actual.interface_values,
        expected.interface_values,
    )
    np.testing.assert_array_equal(
        actual.domain_edges,
        expected.domain_edges,
    )
    np.testing.assert_array_equal(
        actual.domain_index,
        expected.domain_index,
    )


def assert_model_equal(actual, expected):
    scalar_fields = (
        "G",
        "surface_pressure",
        "mass",
        "radius",
        "omega_eq",
    )
    array_fields = (
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

    for name in scalar_fields:
        assert getattr(actual, name) == getattr(expected, name)

    for name in array_fields:
        np.testing.assert_array_equal(
            getattr(actual, name),
            getattr(expected, name),
        )

    assert len(actual.additional_variables) == len(
        expected.additional_variables
    )

    for actual_values, expected_values in zip(
        actual.additional_variables,
        expected.additional_variables,
    ):
        np.testing.assert_array_equal(
            actual_values,
            expected_values,
        )

    assert_domains_equal(actual.domains, expected.domains)


def assert_vacuum_equal(actual, expected):
    scalar_fields = (
        "G",
        "mass",
        "radius",
        "omega_eq",
    )
    array_fields = (
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

    for name in scalar_fields:
        assert getattr(actual, name) == getattr(expected, name)

    for name in array_fields:
        np.testing.assert_array_equal(
            getattr(actual, name),
            getattr(expected, name),
        )

    assert_domains_equal(actual.domains, expected.domains)


def assert_info_equal(actual, expected):
    assert actual.method == expected.method
    assert actual.iterations == expected.iterations
    assert actual.tolerance == expected.tolerance
    assert actual.error == expected.error
    assert actual.rotation_target == expected.rotation_target
    assert actual.elapsed_time == expected.elapsed_time

    np.testing.assert_array_equal(
        actual.polar_radius_history,
        expected.polar_radius_history,
    )


@pytest.mark.parametrize(
    "with_vacuum",
    [
        False,
        True,
    ],
)
def test_hdf5_result_round_trip(tmp_path, with_vacuum):
    model = make_model()
    vacuum = make_vacuum() if with_vacuum else None
    info = make_info()
    path = tmp_path / "model.h5"

    save_result(
        path,
        model,
        vacuum,
        info,
    )
    loaded_model, loaded_vacuum, loaded_info = load_result(path)

    assert_model_equal(loaded_model, model)
    assert_info_equal(loaded_info, info)

    if vacuum is None:
        assert loaded_vacuum is None
    else:
        assert_vacuum_equal(loaded_vacuum, vacuum)


def test_hdf5_result_contains_versioned_groups(tmp_path):
    path = tmp_path / "model.h5"

    save_result(
        path,
        make_model(),
        make_vacuum(),
        make_info(),
    )

    with h5py.File(path, "r") as file:
        assert file.attrs["format_name"] == "RUBIS result"
        assert file.attrs["format_version"] == "1.0"
        assert isinstance(file.attrs["rubis_version"], str)
        assert isinstance(file.attrs["created_at"], str)

        assert set(file) == {
            "model",
            "solver",
            "vacuum",
        }


def test_hdf5_result_does_not_overwrite_by_default(tmp_path):
    path = tmp_path / "model.h5"

    save_result(
        path,
        make_model(),
        None,
        make_info(),
    )

    with pytest.raises(FileExistsError):
        save_result(
            path,
            make_model(),
            None,
            make_info(),
        )


def test_hdf5_result_can_be_overwritten_explicitly(tmp_path):
    path = tmp_path / "model.h5"

    save_result(
        path,
        make_model(),
        None,
        make_info(),
    )
    save_result(
        path,
        make_model(),
        make_vacuum(),
        make_info(),
        overwrite=True,
    )

    _, vacuum, _ = load_result(path)

    assert vacuum is not None


def make_solver_options():
    return SolverOptions(
        method="spheroidal",
        max_degree=31,
        angular_resolution=29,
        full_rate=2,
        mapping_precision=2.0e-11,
        spline_order=3,
        lagrange_order=2,
        external_domain_res=41,
        rescale_ab=False,
        max_iterations=73,
        verbose=True,
    )


def make_virial_balance():
    return VirialBalance(
        kinetic_energy=1.2,
        potential_work=3.4,
        thermodynamic_work=-0.7,
        surface_work=0.05,
    )


def make_gravitational_moments():
    return GravitationalMoments(
        degrees=np.array([
            0,
            2,
            4,
            6,
        ]),
        values=np.array([
            1.0,
            -1.2e-2,
            3.4e-4,
            -5.6e-6,
        ]),
    )


def assert_diagnostics_equal(actual, expected):
    assert (
        actual.virial_balance
        == expected.virial_balance
    )

    actual_moments = actual.gravitational_moments
    expected_moments = expected.gravitational_moments

    if expected_moments is None:
        assert actual_moments is None
        return

    np.testing.assert_array_equal(
        actual_moments.degrees,
        expected_moments.degrees,
    )
    np.testing.assert_array_equal(
        actual_moments.values,
        expected_moments.values,
    )


def test_hdf5_solver_options_round_trip(tmp_path):
    path = tmp_path / "model.h5"
    options = make_solver_options()

    save_result(
        path,
        make_model(),
        None,
        make_info(),
        solver_options=options,
    )

    assert load_solver_options(path) == options

    with h5py.File(path, "r") as file:
        assert "options" in file["solver"]
        assert "diagnostics" not in file


@pytest.mark.parametrize(
    (
        "with_virial",
        "with_moments",
    ),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
def test_hdf5_diagnostics_round_trip(
    tmp_path,
    with_virial,
    with_moments,
):
    path = tmp_path / "model.h5"
    diagnostics = ModelDiagnostics(
        virial_balance=(
            make_virial_balance()
            if with_virial
            else None
        ),
        gravitational_moments=(
            make_gravitational_moments()
            if with_moments
            else None
        ),
    )

    save_result(
        path,
        make_model(),
        None,
        make_info(),
        diagnostics=diagnostics,
    )
    loaded = load_diagnostics(path)

    assert_diagnostics_equal(
        loaded,
        diagnostics,
    )

    with h5py.File(path, "r") as file:
        group = file["diagnostics"]

        assert (
            "virial_balance" in group
        ) == with_virial
        assert (
            "gravitational_moments" in group
        ) == with_moments


def test_hdf5_optional_data_are_absent_by_default(tmp_path):
    path = tmp_path / "model.h5"

    save_result(
        path,
        make_model(),
        None,
        make_info(),
    )

    assert load_solver_options(path) is None
    assert load_diagnostics(path) is None

    with h5py.File(path, "r") as file:
        assert "options" not in file["solver"]
        assert "diagnostics" not in file


def test_hdf5_empty_diagnostics_are_not_written(tmp_path):
    path = tmp_path / "model.h5"

    save_result(
        path,
        make_model(),
        None,
        make_info(),
        diagnostics=ModelDiagnostics(),
    )

    assert load_diagnostics(path) is None

    with h5py.File(path, "r") as file:
        assert "diagnostics" not in file
