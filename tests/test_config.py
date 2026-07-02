from pathlib import Path

import numpy as np

from rubis.config import (
    CompositePolytropeConfig,
    DeformationConfig,
    LegacyModelConfig,
    PolytropeConfig,
    RadiativeFluxOptions,
    RotationConfig,
    SolverOptions,
)
from rubis.rotation_profiles import solid
    
    
def test_radiative_flux_options_are_independent():
    options = RadiativeFluxOptions()

    assert options.origin == 0.05
    assert options.n_lines == 15
    assert options.max_degree is None
    assert options.spline_order == 5
    

def test_polytrope_config_has_one_region():
    config = PolytropeConfig(index=3.0)

    assert config.polytropic_indices == (3.0,)
    assert config.n_regions == 1
    assert config.filename_stem == "poly_|3.0|"

    assert config.radius == 1.0
    assert config.mass == 1.0
    assert config.n_points == 1001


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


def test_legacy_model_config_builds_path(tmp_path):
    config = LegacyModelConfig(
        filename="model.txt",
        directory=tmp_path,
    )

    assert config.path == tmp_path / "model.txt"
    assert config.filename_stem == "model"


def test_deformation_config_builds_default_subconfigs():
    model = PolytropeConfig(index=1.0)
    config = DeformationConfig(model=model)

    assert config.model is model
    assert isinstance(config.rotation, RotationConfig)
    assert isinstance(config.solver, SolverOptions)

    assert config.rotation.profile is solid
    assert config.rotation.target == 0.0
    assert config.solver.method == "auto"