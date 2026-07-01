from .numerical import interpolate_func


def integrate_pressure(
    zeta,
    rho,
    dphi_eff,
    surface_pressure,
    *,
    unique_indices=None,
    spline_order=3,
):
    """
    Integrate hydrostatic equilibrium from the surface inward.

    Duplicated material interfaces may be excluded from the interpolation
    while retaining both copies in the reconstructed pressure profile.
    """
    dp = -rho * dphi_eff
    x = 1.0 - zeta[::-1]

    if unique_indices is None:
        x_ipl = x
        dp_ipl = dp[::-1]
    else:
        x_ipl = 1.0 - zeta[unique_indices][::-1]
        dp_ipl = dp[unique_indices][::-1]

    p = interpolate_func(
        x=x_ipl,
        y=-dp_ipl,
        der=-1,
        k=spline_order,
        prim_cond=(0, surface_pressure),
    )(x)

    return p[::-1]