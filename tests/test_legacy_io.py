import numpy as np

from rubis.io.legacy import write_model


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