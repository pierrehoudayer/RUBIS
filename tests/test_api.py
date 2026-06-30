import numpy as np
import pytest

import rubis.api as api
from rubis.config import (
    DeformationConfig,
    RotationConfig,
    SolverOptions,
)
from rubis.models import (
    CompositePolytropeConfig,
    PolytropeConfig,
)
from rubis.rotation_profiles import solid


def test_auto_selects_radial_method_for_single_region(monkeypatch):
    expected = object()

    monkeypatch.setattr(
        api,
        "radial_method",
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


def test_auto_selects_spheroidal_method_for_multiple_regions(
    monkeypatch,
):
    expected = object()

    monkeypatch.setattr(
        api,
        "radial_method",
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
            model=CompositePolytropeConfig(
                indices=(1.0, 1.0),
                target_pressures=(-1.0, -np.inf),
                density_jumps=(0.4,),
            ),
        )
    )

    assert result is expected


def test_explicit_method_overrides_automatic_selection(
    monkeypatch,
):
    expected = object()

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
    calls = {}
    expected = object()

    def fake_solver(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(
        api,
        "radial_method",
        fake_solver,
    )

    model = PolytropeConfig(index=1.0)
    rotation = RotationConfig(
        profile=solid,
        target=0.3,
        central_diff_rate=0.2,
        scale=0.7,
    )
    solver = SolverOptions(
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
    )

    config = DeformationConfig(
        model=model,
        rotation=rotation,
        solver=solver,
    )

    result = api.deform(config)

    assert result is expected
    assert calls["args"] == (
        model,
        solid,
        0.3,
        0.2,
        0.7,
        9,
        11,
        2,
        1.0e-8,
        3,
        2,
        config.output,
        21,
        False,
    )
    assert calls["kwargs"] == {
        "max_iterations": 17,
    }