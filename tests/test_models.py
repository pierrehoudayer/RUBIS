import numpy as np

from rubis.models import (
    CompositePolytropeConfig,
    PolytropeConfig,
    SphericalModel,
)


def test_polytrope_config_has_one_region():
    config = PolytropeConfig(index=3.0)

    assert config.polytropic_indices == (3.0,)
    assert config.n_regions == 1
    assert config.filename_stem == "poly_|3.0|"


def test_composite_polytrope_config_accepts_scalar_index():
    config = CompositePolytropeConfig(
        indices=3.0,
        target_pressures=-np.inf,
    )

    assert config.polytropic_indices == (3.0,)
    assert config.n_regions == 1


def test_composite_polytrope_config_has_multiple_regions():
    config = CompositePolytropeConfig(
        indices=(1.5, 3.0),
        target_pressures=(-1.0, -np.inf),
        density_jumps=(0.5,),
    )

    assert config.polytropic_indices == (1.5, 3.0)
    assert config.n_regions == 2
    assert config.filename_stem == "poly_|1.5|3.0|"


def test_spherical_model_reports_number_of_points():
    model = SphericalModel(
        r=np.linspace(0.0, 1.0, 5),
        p=np.ones(5),
        rho=np.ones(5),
        g=np.ones(5),
    )

    assert model.n_points == 5