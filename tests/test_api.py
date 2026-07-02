import numpy as np
import pytest
from types import SimpleNamespace

import rubis.api as api
from rubis.config import (
    CompositePolytropeConfig,
    DeformationConfig,
    OutputOptions,
    PolytropeConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.domains import find_domains
from rubis.models import Model1D
from rubis.rotation_profiles import solid


def make_model_1d(*, multidomain=False):
    if multidomain:
        r = np.array([0.0, 0.5, 0.5, 1.0])
    else:
        r = np.array([0.0, 0.5, 1.0])

    return Model1D(
        G=6.67384e-8,
        surface_pressure=0.0,
        mass=1.0,
        radius=1.0,
        r=r,
        rho=np.ones_like(r),
        domains=find_domains(r),
    )
    
    
def test_auto_selects_radial_solver(monkeypatch):
    model = make_model_1d()
    expected = object()

    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: model,
    )
    monkeypatch.setattr(
        api,
        "solve_radial",
        lambda *args, **kwargs: expected,
    )
    monkeypatch.setattr(
        api,
        "solve_spheroidal",
        lambda *args, **kwargs: pytest.fail(
            "The spheroidal solver should not be called."
        ),
    )

    result = api.deform(
        DeformationConfig(
            model=PolytropeConfig(index=1.0),
        )
    )

    assert result is expected
    
    
def test_auto_selects_spheroidal_solver(monkeypatch):
    model = make_model_1d(multidomain=True)
    expected = object()

    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: model,
    )
    monkeypatch.setattr(
        api,
        "solve_radial",
        lambda *args, **kwargs: pytest.fail(
            "The radial solver should not be called."
        ),
    )
    monkeypatch.setattr(
        api,
        "solve_spheroidal",
        lambda *args, **kwargs: expected,
    )

    result = api.deform(
        DeformationConfig(
            model=PolytropeConfig(index=1.0),
        )
    )

    assert result is expected
    
    
def test_explicit_method_overrides_domain_selection(
    monkeypatch,
):
    model = make_model_1d()
    expected = object()

    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: model,
    )
    monkeypatch.setattr(
        api,
        "solve_spheroidal",
        lambda *args, **kwargs: expected,
    )

    result = api.deform(
        DeformationConfig(
            model=PolytropeConfig(index=1.0),
            solver=SolverOptions(
                method="spheroidal",
            ),
        )
    )

    assert result is expected
    
    
def test_deform_forwards_configuration(monkeypatch):
    model1d = object()
    model2d = object()
    info = object()
    solver_output = (model2d, None, info)

    calls = {}

    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: model1d,
    )

    def fake_solve_radial(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return solver_output

    monkeypatch.setattr(
        api,
        "solve_radial",
        fake_solve_radial,
    )

    config = DeformationConfig(
        model=PolytropeConfig(
            index=1.0,
        ),
        rotation=RotationConfig(
            profile=solid,
            target=0.1,
        ),
        solver=SolverOptions(
            method="radial",
        ),
        output=OutputOptions(),
    )

    result = api.deform(config)

    assert result is solver_output
    assert calls["args"] == (
        model1d,
        config.rotation,
        config.solver,
        config.output,
    )
    assert calls["kwargs"] == {}
    
    
def test_deform_rejects_unknown_method(monkeypatch):
    model = make_model_1d()

    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: model,
    )

    config = DeformationConfig(
        model=PolytropeConfig(index=1.0),
        solver=SolverOptions(
            method="unknown",  # type: ignore[arg-type]
        ),
    )

    with pytest.raises(
        ValueError,
        match="Unknown deformation method",
    ):
        api.deform(config)
        
        
def test_deform_selects_solver_from_model_domains(
    monkeypatch,
):
    radial_output = (
        object(),
        None,
        object(),
    )
    spheroidal_output = (
        object(),
        object(),
        object(),
    )

    calls = []

    def fake_solve_radial(*args):
        calls.append("radial")
        return radial_output

    def fake_solve_spheroidal(*args):
        calls.append("spheroidal")
        return spheroidal_output

    monkeypatch.setattr(
        api,
        "solve_radial",
        fake_solve_radial,
    )
    monkeypatch.setattr(
        api,
        "solve_spheroidal",
        fake_solve_spheroidal,
    )

    radial_model = SimpleNamespace(
        n_domains=1,
    )
    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: radial_model,
    )

    radial_config = DeformationConfig(
        model=PolytropeConfig(index=1.0),
        rotation=RotationConfig(
            profile=solid,
        ),
        solver=SolverOptions(
            method="auto",
        ),
        output=OutputOptions(),
    )

    assert api.deform(radial_config) is radial_output
    assert calls == ["radial"]

    spheroidal_model = SimpleNamespace(
        n_domains=2,
    )
    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        lambda config: spheroidal_model,
    )

    spheroidal_config = DeformationConfig(
        model=CompositePolytropeConfig(
            indices=(1.0, 1.0),
            target_pressures=(-1.0, -np.inf),
            density_jumps=(0.4,),
        ),
        rotation=RotationConfig(
            profile=solid,
        ),
        solver=SolverOptions(
            method="auto",
        ),
        output=OutputOptions(),
    )

    assert (
        api.deform(spheroidal_config)
        is spheroidal_output
    )
    assert calls == [
        "radial",
        "spheroidal",
    ]