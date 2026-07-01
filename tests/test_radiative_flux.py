import numpy as np

import rubis.solvers.radial as radial
from rubis.config import RadiativeFluxOptions, SolverOptions
from rubis.mapping import initialize_mapping
from rubis.solvers.radial import (
    find_radiative_flux,
    initialize_radial_numerics,
)


def test_radiative_flux_is_uniform_for_spherical_mapping(monkeypatch):
    zeta = np.linspace(0.0, 1.0, 41)

    options = SolverOptions(
        method="radial",
        max_degree=11,
        angular_resolution=21,
        spline_order=3,
        lagrange_order=2,
    )

    r2d, t = initialize_mapping(
        zeta,
        options.angular_resolution,
    )

    num = initialize_radial_numerics(
        zeta,
        t,
        options,
    )

    # Avoid opening or constructing a plot during the test.
    monkeypatch.setattr(
        radial,
        "plot_3D_surface",
        lambda *args, **kwargs: None,
    )

    Q_l, (fig, ax) = find_radiative_flux(
        r2d,
        zeta,
        num,
        RadiativeFluxOptions(
            origin=0.2,
            n_lines=8,
            plot_lines=False,
        ),
    )

    assert np.all(np.isfinite(Q_l))
    assert Q_l[0] > 0.0
    assert np.linalg.norm(Q_l[1:]) < 1.0e-6 * abs(Q_l[0])

    assert fig is None
    assert ax is None