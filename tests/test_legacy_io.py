import numpy as np

from rubis.config import (
    CompositePolytropeConfig,
    PolytropeConfig,
)
from rubis.io.legacy import make_output_filename, write_model


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