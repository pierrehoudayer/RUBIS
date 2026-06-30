import numpy as np

from rubis.config import (
    CompositePolytropeConfig,
    LegacyModelConfig,
    PolytropeConfig,
)
from rubis.initialization import G, initialize_model_1d
from rubis.polytrope import build_polytrope


def test_initialize_polytrope_normalises_model():
    config = PolytropeConfig(
        index=1.0,
        radius=2.0,
        mass=3.0,
        n_points=101,
    )

    source = build_polytrope(config)
    model = initialize_model_1d(config)

    np.testing.assert_allclose(
        model.r,
        source.r / config.radius,
    )
    np.testing.assert_allclose(
        model.rho,
        source.rho / (
            config.mass / config.radius**3
        ),
    )
    np.testing.assert_allclose(
        model.surface_pressure,
        source.p[-1] / (
            G * config.mass**2 / config.radius**4
        ),
    )

    assert model.G == G
    assert model.mass == config.mass
    assert model.radius == config.radius
    assert model.n_points == config.n_points
    assert model.n_domains == 1
    assert model.additional_variables == ()
    
    
def test_initialize_composite_model_preserves_domains():
    config = CompositePolytropeConfig(
        indices=(1.0, 1.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.4,),
        n_points=101,
    )

    model = initialize_model_1d(config)

    assert model.n_domains == 2
    assert model.domains.has_interfaces

    duplicated = np.flatnonzero(
        np.diff(model.r) == 0.0
    )
    assert duplicated.size == 1
    
    
def test_initialize_legacy_model_reads_additional_variables(
    tmp_path,
):
    path = tmp_path / "model.txt"

    r = np.linspace(0.0, 1.0, 9)
    rho = np.ones_like(r)
    temperature = np.linspace(10.0, 90.0, r.size)

    with path.open("w") as file:
        file.write(f"0.0\n{r.size}\n")
        np.savetxt(
            file,
            np.column_stack((
                r,
                rho,
                temperature,
            )),
        )

    config = LegacyModelConfig(
        filename=path.name,
        directory=tmp_path,
    )

    model = initialize_model_1d(config)

    assert model.n_points == r.size
    assert model.n_domains == 1
    assert len(model.additional_variables) == 1

    np.testing.assert_allclose(
        model.r,
        r,
    )
    np.testing.assert_allclose(
        model.rho,
        rho / model.mass,
    )
    np.testing.assert_array_equal(
        model.additional_variables[0],
        temperature,
    )