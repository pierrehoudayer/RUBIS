import numpy as np
import pytest

import rubis.api as api
from rubis.config import (
    DeformationConfig,
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
        "spheroidal_method",
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
        "spheroidal_method",
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
        "spheroidal_method",
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
    model = make_model_1d()
    calls = {}
    expected = object()

    def fake_initialize(config):
        calls["model_config"] = config
        return model

    def fake_solver(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        api,
        "initialize_model_1d",
        fake_initialize,
    )
    monkeypatch.setattr(
        api,
        "solve_radial",
        fake_solver,
    )

    model_config = PolytropeConfig(index=1.0)

    config = DeformationConfig(
        model=model_config,
        rotation=RotationConfig(
            profile=solid,
            target=0.3,
            central_diff_rate=0.2,
            scale=0.7,
        ),
        solver=SolverOptions(
            method="radial",
            max_degree=9,
            angular_resolution=11,
            full_rate=2,
            mapping_precision=1.0e-8,
            spline_order=3,
            lagrange_order=2,
            external_domain_res=21,
            rescale_ab=False,
            max_iterations=17,
        ),
    )

    result = api.deform(config)

    assert result is expected
    assert calls["model_config"] is model_config

    assert calls["args"][0] is model
    assert calls["args"][1] is config.rotation
    assert calls["args"][2] is config.solver
    assert calls["args"][3] is config.output
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