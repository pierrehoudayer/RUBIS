import numpy as np

from rubis.config import (
    CompositePolytropeConfig,
    PolytropeConfig,
)
from rubis.io.legacy import (
    make_output_filename,
    write_deformed_model,
    write_model,
)

def test_make_output_filename_from_model_file():
    filename = make_output_filename(
        "solar_model.txt",
        0.9,
    )

    assert filename == "solar_model_deform_0.9.txt"


def test_make_output_filename_from_single_polytrope():
    model = PolytropeConfig(index=3.0)

    filename = make_output_filename(model, 0.7)

    assert filename == "poly_|3.0|_deform_0.7.txt"
    
    
def test_make_output_filename_from_composite_polytrope():
    model = CompositePolytropeConfig(
        indices=(1.0, 1.5),
        target_pressures=(-1.0, -np.inf),
    )

    filename = make_output_filename(model, 0.8)

    assert filename == "poly_|1.0|1.5|_deform_0.8.txt"


def test_write_model_preserves_legacy_format(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models").mkdir()

    mapping = np.array([
        [0.0, 0.0],
        [0.5, 0.6],
        [1.0, 1.2],
    ])
    zeta = np.array([0.0, 0.5, 1.0])
    rho = np.array([2.0, 1.0, 0.0])
    composition = np.array([0.7, 0.7, 0.7])

    params = (
        3,
        2,
        1.5,
        2.0,
        0.3,
        6.67e-8,
    )

    write_model(
        "test_model.txt",
        params,
        mapping,
        (composition,),
        zeta,
        rho,
    )

    path = tmp_path / "Models" / "test_model.txt"

    first_line = path.read_text().splitlines()[0]
    assert first_line == " ".join(str(value) for value in params)

    data = np.loadtxt(path, skiprows=1)
    expected = np.column_stack((
        mapping,
        zeta,
        rho,
        composition,
    ))

    np.testing.assert_allclose(
        data,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    
    
def test_write_deformed_model_applies_dimensional_scales(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Models").mkdir()

    r2d = np.array([
        [0.0, 0.0, 0.0],
        [0.4, 0.5, 0.4],
        [0.8, 1.0, 0.8],
    ])
    zeta = np.array([0.0, 0.5, 1.0])
    p = np.array([3.0, 2.0, 1.0])
    rho = np.array([2.0, 1.5, 1.0])
    phi_eff = np.array([-2.0, -1.5, -1.0])
    omega_equator = np.array([0.2, 0.2, 0.2])
    additional = (np.array([10.0, 20.0, 30.0]),)

    mass = 2.0
    radius = 4.0
    G = 5.0

    write_deformed_model(
        "model.txt",
        r2d=r2d,
        additional_variables=additional,
        zeta=zeta,
        p=p,
        rho=rho,
        phi_eff=phi_eff,
        omega_equator=omega_equator,
        mass=mass,
        radius=radius,
        rotation_target=0.3,
        G=G,
        dimensional=True,
    )

    data = np.loadtxt(
        tmp_path / "Models" / "model.txt",
        skiprows=1,
    )

    J = r2d.shape[1]

    np.testing.assert_allclose(
        data[:, :J],
        r2d * radius,
    )
    np.testing.assert_allclose(
        data[:, J + 0],
        zeta,
    )
    np.testing.assert_allclose(
        data[:, J + 1],
        p * G * mass**2 / radius**4,
    )
    np.testing.assert_allclose(
        data[:, J + 2],
        rho * mass / radius**3,
    )
    np.testing.assert_allclose(
        data[:, J + 3],
        phi_eff * G * mass / radius,
    )
    np.testing.assert_allclose(
        data[:, J + 4],
        omega_equator,
    )
    np.testing.assert_allclose(
        data[:, J + 5],
        additional[0],
    )