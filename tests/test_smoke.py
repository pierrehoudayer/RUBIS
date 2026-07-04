def test_core_modules_import():
    import rubis.api as api
    import rubis.config as config
    import rubis.diagnostics as diagnostics
    import rubis.domains as domains
    import rubis.flux as flux
    import rubis.hydrostatics as hydrostatics
    import rubis.initialization as initialization
    import rubis.lagrange as lagrange
    import rubis.legendre as legendre
    import rubis.mapping as mapping
    import rubis.models as models
    import rubis.plotting as plotting
    import rubis.poisson as poisson
    import rubis.polytrope as polytrope
    import rubis.quadrature as quadrature
    import rubis.results as results
    import rubis.rotation_profiles as rotation_profiles
    import rubis.rotation as rotation
    import rubis.special as special
    
def test_rubis_package_import():
    import rubis

    assert rubis.__version__ == "2.0.0"


def test_public_api_is_exposed_from_package():
    import rubis
    from rubis.api import deform
    from rubis.config import (
        CompositePolytropeConfig,
        DeformationConfig,
        LegacyModelConfig,
        PolytropeConfig,
        RadiativeFluxOptions,
        RotationConfig,
        SolverOptions,
    )
    from rubis.models import Model2D, VacuumModel2D
    from rubis.results import SolverInfo, SolverOutput

    assert rubis.deform is deform
    assert rubis.CompositePolytropeConfig is CompositePolytropeConfig
    assert rubis.DeformationConfig is DeformationConfig
    assert rubis.LegacyModelConfig is LegacyModelConfig
    assert rubis.Model2D is Model2D
    assert rubis.PolytropeConfig is PolytropeConfig
    assert rubis.RadiativeFluxOptions is RadiativeFluxOptions
    assert rubis.RotationConfig is RotationConfig
    assert rubis.SolverInfo is SolverInfo
    assert rubis.SolverOutput is SolverOutput
    assert rubis.SolverOptions is SolverOptions
    assert rubis.VacuumModel2D is VacuumModel2D
