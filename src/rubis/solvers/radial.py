import time
import numpy        as np
import scipy.sparse as sps
from dataclasses         import dataclass
from numpy.typing        import NDArray
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.special       import roots_legendre
from scipy.integrate     import solve_ivp

from ..config            import (
    OutputOptions,
    RadiativeFluxOptions,
    RotationConfig, 
    SolverOptions,
)
from ..legendre          import (
    find_r_eq, 
    find_r_pol, 
    pl_eval_2D, 
    pl_project_2D,
)
from ..numerical         import (
    integrate, 
    integrate2D, 
    interpolate_func, 
    lagrange_matrix_P,
)
from ..models            import Model1D
from ..mapping           import (
    initialize_mapping, 
    valid_reciprocal_domain,
    compute_mapping_derivatives,
    compute_mapping_geometry,
)
from ..hydrostatics      import integrate_pressure
from ..rotation          import (
    RotationState,
    initialize_rotation_state,
)
from ..results           import RadialResult
from ..diagnostics       import (
    compute_gravitational_moments,
    compute_virial_balance,
    report_gravitational_moments,
    report_virial_balance,
)
from ..io.legacy         import write_deformed_model
from ..plotting          import (
    phi_g_harmonics,
    plot_3D_surface,
    plot_f_map,
    plot_flux_lines,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class RadialNumerics:
    """
    Fixed numerical representation used by the radial solver.

    It stores the spherical radial grid, the angular collocation grid,
    and the discrete operators that remain unchanged during iteration.
    """

    r1d: FloatArray
    t: FloatArray

    max_degree: int
    spline_order: int
    lagrange_order: int

    Lsp: sps.spmatrix
    Dsp: sps.spmatrix
    Asp: sps.spmatrix

    @property
    def n_points(self) -> int:
        return self.r1d.size

    @property
    def angular_resolution(self) -> int:
        return self.t.size


def initialize_radial_numerics(
    r1d,
    t,
    options: SolverOptions,
) -> RadialNumerics:
    """
    Build the fixed grids and operators of the radial solver.

    The radial operators are constructed on the spherical Poisson grid,
    independently of the evolving material mapping.
    """
    lag_mat = lagrange_matrix_P(
        r1d**2,
        order=options.lagrange_order,
    )

    Lsp = sps.dia_matrix(lag_mat[..., 0])
    Dsp = sps.dia_matrix(lag_mat[..., 1])
    Asp = sps.dia_matrix(
          4 * lag_mat[..., 1] * r1d**4
        - 2 * lag_mat[..., 0] * r1d**2
    )

    return RadialNumerics(
        r1d=r1d,
        t=t,
        max_degree=options.max_degree,
        spline_order=options.spline_order,
        lagrange_order=options.lagrange_order,
        Lsp=Lsp,
        Dsp=Dsp,
        Asp=Asp,
    )


def compute_density_harmonics(
    r2d,
    rho,
    num: RadialNumerics,
):
    """
    Reconstruct the density on spherical shells and project it onto
    Legendre harmonics.

    Density is attached to the material surfaces of the current mapping
    and is interpolated onto the fixed spherical Poisson grid.
    """
    safety_constant = 1.0e-15
    L = num.max_degree
    spl_order = num.spline_order
    
    J = r2d.shape[1]
    j_dw = np.arange((J + 1) // 2)
    j_up = -j_dw - 1

    r = num.r1d
    log_rho = np.log(rho + safety_constant)
    rho2D = np.zeros_like(r2d)

    for j in j_dw:
        inside = r < r2d[-1, j]
        rho2D[inside, j] = interpolate_func(x=r2d[:, j], y=log_rho, k=spl_order)(r[inside])
        rho2D[inside, j] = np.exp(rho2D[inside, j]) - safety_constant

    rho2D[:, j_up] = rho2D[:, j_dw]

    return pl_project_2D(rho2D, L)
    
    
def fill_poisson_band_matrix(
    ab, 
    ku, 
    kl, 
    l, 
    num: RadialNumerics
):
    """
    Fill the banded Poisson operator for one Legendre degree.

    The degree-independent radial terms are initialized for the monopole
    and reused while the harmonic and boundary terms are updated.
    """
    I = num.n_points
    offset = ku + kl

    # Common part, filled only once
    if l == 0:
        ab[ku+1+(0-0):-1+(0-0):2, 0::2] =  num.Asp.data[::-1]
        ab[ku+1+(1-0):        :2, 0::2] = -num.Lsp.data[::-1]
        ab[ku+1+(1-1):-1+(1-1):2, 1::2] =  num.Dsp.data[::-1]

        # Central boundary condition for l = 0
        ab[offset, 0] = 6.0

    # Degree-dependent part
    else:
        ab[ku+1+(0-1):-1+(0-1):2, 1::2] = -l*(l+1) * num.Lsp.data[::-1]

        # Central boundary condition for l != 0
        ab[offset-0, 0] = 0.0
        ab[offset-1, 1] = 1.0

    # Surface boundary conditions
    ab[offset+1, 2*I-2] = 2*num.r1d[-1]**2
    ab[offset+0, 2*I-1] = l+1 

    return ab

    
def solve_gravitational_potential(
    r2d,
    rho,
    num: RadialNumerics,
    phi_eff=None,
    poisson_factors=None,
):
    """
    Solve Poisson's equation on the fixed spherical radial grid.

    The density is first reconstructed and projected onto Legendre
    harmonics, after which each even degree is solved independently.
    The effective-potential profile is initialized on the first call
    and subsequently shifted to remain consistent at the centre.
    """
    I = num.n_points
    L = num.max_degree
    r1d = num.r1d[:, None]
    
    # Density distribution harmonics
    rho_l    = compute_density_harmonics(r2d, rho, num)
    phi_g_l  = np.zeros((I, L))
    dphi_g_l = np.zeros((I, L))
    
    # Vector filling (vectorial)
    b_l = np.zeros((2*I, (L + 1) // 2))
    b_l[1:-1:2, :] = 4 * np.pi * num.Lsp @ (r1d**2 * rho_l[:, ::2])
    b_l[0     , 0] = 4 * np.pi * rho_l[0, 0]     # Boundary condition
    
    # Band matrix storage
    kl = ku = 2 * num.lagrange_order
    ab = np.zeros((2*kl + ku + 1, 2*I))   
    
    if phi_eff is None :
        poisson_factors = []
        for l in range(0, L, 2) :
            # Matrix filling  
            ab = fill_poisson_band_matrix(ab, ku, kl, l, num)
            
            # LU decomposition (LAPACK)
            poisson_factors.append(dgbtrf(ab, ku, kl)[:-1])
            
    # System solving (LAPACK)
    x = np.array([
        dgbtrs(lu, kl, ku, b, piv)[0]
        for (lu, piv), b in zip(poisson_factors, b_l.T)
    ]).T
        
    # Poisson's equation solution
    phi_g_l[:, ::2]  = x[1::2]
    dphi_g_l[:, ::2] = x[0::2] * (2*r1d)  # <- The equation is solved on r^2
    
    if phi_eff is None :
        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        dphi_eff = pl_eval_2D(dphi_g_l, 0.0)        
        return phi_g_l, dphi_g_l, phi_eff, dphi_eff, poisson_factors
    
    # The effective potential is known up to an additive constant 
    phi_offset = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += phi_offset
    
    return phi_g_l, dphi_g_l, phi_eff


def update_mapping(
    phi_g_l,
    dphi_g_l,
    phi_eff,
    num: RadialNumerics,
    rot: RotationState,
):
    """
    Construct the next material mapping from the total potential.

    The gravitational solution is extended outside the spherical grid,
    combined with the centrifugal potential, and inverted at fixed
    effective potential to recover r(zeta, t). The equatorial rotation
    rate is corrected consistently with the estimated surface radius.
    """
    r = num.r1d
    t = num.t
    L = num.max_degree
    J = num.angular_resolution
    k = num.spline_order

    # Lower angular half-domain, including the equator
    j_eq = (J - 1) // 2
    j_dw = np.arange(j_eq + 1)

    # Interior gravitational potential
    phi2D_g_int  = pl_eval_2D( phi_g_l, t[j_dw])
    dphi2D_g_int = pl_eval_2D(dphi_g_l, t[j_dw])

    # Exterior gravitational potential
    l = np.arange(L)
    outside = 1.3
    r_ext = np.linspace(1.0, outside, 101)[1:]

    phi_g_l_ext = phi_g_l[-1] * r_ext[:, None] ** -(l + 1)
    dphi_g_l_ext = -(l + 1) * phi_g_l_ext / r_ext[:, None]

    phi2D_g_ext  = pl_eval_2D( phi_g_l_ext, t[j_dw])
    dphi2D_g_ext = pl_eval_2D(dphi_g_l_ext, t[j_dw])

    # Full radial domain
    r_tot = np.hstack((r, r_ext))
    phi2D_g  = np.vstack(( phi2D_g_int,  phi2D_g_ext))
    dphi2D_g = np.vstack((dphi2D_g_int, dphi2D_g_ext))

    # Find a rotation rate consistent with the equatorial radius
    safe = r_tot > 0.0
    r_safe = r_tot[safe]

    phi1D_c, dphi1D_c = rot.phi_c(r_safe, 0.0) / r_safe**3
    dphi1D_c -= 3.0 * phi1D_c / r_safe

    phi1D  =  phi2D_g[safe, j_eq] + phi1D_c
    dphi1D = dphi2D_g[safe, j_eq] + dphi1D_c

    r_est = CubicHermiteSpline(x=phi1D, y=r_safe, dydx=dphi1D**-1)(phi_eff[-1])
    omega_eq_new = rot.omega_eq * r_est**-1.5
    rot_new = rot.with_omega_eq(omega_eq_new)

    # Centrifugal potential
    phi2D_c, dphi2D_c = np.moveaxis(
        np.array([
            rot_new.phi_c(r_tot, t_j)
            for t_j in t[j_dw]
        ]),
        (0, 1, 2),
        (2, 0, 1),
    )

    # Total potential
    phi2D  =  phi2D_g + phi2D_c
    dphi2D = dphi2D_g + dphi2D_c

    valid = valid_reciprocal_domain(r_tot, dphi2D)

    # Refined central domain
    lim = 1.0e-1
    lim_idx = np.max(np.argwhere(r_tot < lim)) + 1

    r_cnt = lim * np.linspace(0.0, 1.0, 5 * lim_idx) ** 2

    phi2D_g_cnt = np.array([
        CubicHermiteSpline(
            x=r,
            y=phi2D_g_int[:, j],
            dydx=dphi2D_g_int[:, j],
        )(r_cnt)
        for j in j_dw
    ]).T

    phi2D_c_cnt = np.array([
        rot_new.phi_c(r_cnt, t_j)[0]
        for t_j in t[j_dw]
    ]).T

    phi2D_cnt = phi2D_g_cnt + phi2D_c_cnt

    # Estimate the radius at each target equipotential
    r2d_dw_origin = np.zeros_like(j_dw)
    r2d_dw_center = np.array([
        interpolate_func(x=pk, y=r_cnt, k=k)(phi_eff[1:lim_idx])
        for pk in phi2D_cnt.T
    ]).T
    r2d_dw_envelope = np.array([
        interpolate_func(x=pk[vk], y=r_tot[vk], k=k)(phi_eff[lim_idx:])
        for pk, vk in zip(phi2D.T, valid.T)
    ]).T
    r2d_dw = np.vstack((
        r2d_dw_origin,
        r2d_dw_center,
        r2d_dw_envelope,
    ))
    r2d_up = np.flip(r2d_dw, axis=1)[:, 1:]

    r2d_new = np.hstack((r2d_dw, r2d_up))

    return r2d_new, rot_new
        

def solve_radial(
    model: Model1D,
    rotation_config: RotationConfig,
    solver_options: SolverOptions,
    output_options: OutputOptions,
) -> RadialResult:
    """
    Compute the rotational deformation using a spherical Poisson grid.

    Density remains attached to material equipotential surfaces through
    the mapping r(zeta, t), while Poisson's equation is solved on a fixed
    spherical radial grid. The mapping and rotation state are iterated 
    while the global mass and radius are rescaled to preserve the 
    prescribed model constraints.
    """
    start = time.perf_counter()

    if model.n_domains > 1:
        raise ValueError(
            "The radial solver only supports single-domain models."
        )

    r2d, t = initialize_mapping(
        model.r,
        solver_options.angular_resolution,
    )

    num = initialize_radial_numerics(model.r, t, solver_options)

    G = model.G
    surface_pressure = model.surface_pressure
    radius = model.radius
    mass = model.mass
    additional_variables = model.additional_variables

    r1d  = num.r1d.copy()
    zeta = num.r1d.copy()
    rho = model.rho.copy()

    L = num.max_degree
    spl_order = num.spline_order

    rot = initialize_rotation_state(rotation_config)
    rotation_target = rotation_config.target
    
    full_rate = solver_options.full_rate
    mapping_precision = solver_options.mapping_precision
    max_iterations = solver_options.max_iterations
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff, poisson_factors = solve_gravitational_potential(r2d, rho, num)
    
    # Find pressure
    p = integrate_pressure(
        zeta,
        rho,
        dphi_eff,
        surface_pressure,
        spline_order=num.spline_order,
    )
    
    # Iterative centrifugal deformation
    polar_radius_history = [0.0, find_r_pol(r2d, L)]
    iterations = 0
    print(
        "\n+---------------------+",
        "\n| Deformation started |", 
        "\n+---------------------+\n"
    )    
    
    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be a positive integer."
        )
    
    while abs(polar_radius_history[-1] - polar_radius_history[-2]) > mapping_precision:
        if iterations >= max_iterations:
            delta_polar = abs(
                polar_radius_history[-1] - polar_radius_history[-2]
            )

            recent_radii = np.asarray(
                polar_radius_history[-4:]
            )

            raise RuntimeError(
                "Radial deformation did not converge after "
                f"{max_iterations} iterations. "
                f"Last |delta R_pol| = "
                f"{delta_polar:.3e}, "
                f"target = {mapping_precision:.3e}. "
                f"Recent polar radii: "
                f"{recent_radii!r}"
            )
        
        # Current rotation rate
        rotation_cap = ((iterations+1)/full_rate) * rotation_target
        rot = rot.with_omega_eq(min(rotation_target, rotation_cap))
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = solve_gravitational_potential(
            r2d,
            rho,
            num,
            phi_eff=phi_eff,
            poisson_factors=poisson_factors,
        )

        # Find a new estimate for the mapping
        r2d, rot = update_mapping(
            phi_g_l,
            dphi_g_l,
            phi_eff,
            num,
            rot,
        )
        
        # Renormalisation
        r_corr    = find_r_eq(r2d, L)
        m_corr    = integrate2D(r2d, rho, k=spl_order)
        radius   *= r_corr
        mass     *= m_corr
        r2d      /=             r_corr
        rho      /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr**2
        p        /= m_corr**2 / r_corr**4
        
        # Update the polar radius
        polar_radius_history.append(find_r_pol(r2d, L))
        
        # Iteration count
        iterations += 1
        n_decimals = int(-np.log10(mapping_precision))
        print(
            f"Iteration n°{iterations:02d}:",
            f"R_pol = {polar_radius_history[-1].round(n_decimals)}"
        )
    
    # Deformation summary
    finish = time.perf_counter()
    print(
        "\n+------------------+",
        "\n| Deformation done |", 
        "\n+------------------+\n"
    )
    print(f'Time taken: {round(finish-start, 2)} secs')  
    
    
    # Store the normalised solver state before any output
    # operation can modify the arrays in place
    result = RadialResult(
        zeta=zeta.copy(),
        radial_grid=r1d.copy(),
        cos_theta=t.copy(),
        mapping=r2d.copy(),

        density=rho.copy(),
        pressure=p.copy(),

        effective_potential=phi_eff.copy(),
        effective_potential_derivative=dphi_eff.copy(),

        gravitational_potential_harmonics=phi_g_l.copy(),
        gravitational_potential_derivative_harmonics=dphi_g_l.copy(),

        mass=mass,
        radius=radius,

        rotation_target=rotation_target,
        rotation_rate=rot.omega_eq,

        polar_radius_history=np.asarray(polar_radius_history),
        iterations=iterations,
    )

    # Gravitational-potential harmonics
    if output_options.plot.show_harmonics : 
        phi_g_harmonics(zeta, phi_g_l, radial=True)
    
    # Virial test
    if output_options.diagnostics.virial_test:
        phi_g2d = (
            phi_eff[:, None]
            - rot.phi_c2d(r2d, num.t)
        )

        balance = compute_virial_balance(
            r2d,
            rho,
            p,
            phi_g2d,
            num.t,
            rot,
            spline_order=num.spline_order,
        )

        report_virial_balance(
            balance,
            verbose=True,
        )
    
    # Plot model
    if output_options.plot.show_model :
        
        # Variable to plot
        f = rho
        label = r"$\rho \times {\left(M/R_{\mathrm{eq}}^3\right)}^{-1}$"
            
        if output_options.flux.enabled:
            Q_l, (fig, ax) = compute_radiative_flux(
                r2d,
                zeta,
                num,
                output_options.flux,
            )

            plot_f_map(
                r2d,
                f,
                phi_eff,
                L,
                angular_res=output_options.plot.resolution,
                cmap=output_options.plot.field_cmap,
                show_surfaces=output_options.plot.surfaces,
                cmap_lines=output_options.plot.surface_cmap,
                label=label,
                add_to_fig=(
                    (fig, ax)
                    if output_options.flux.plot_lines
                    else None
                ),
            )
    
    # Gravitational moments
    if output_options.diagnostics.gravitational_moments:
        moments = compute_gravitational_moments(
            r2d,
            rho,
            num.t,
            spline_order=num.spline_order,
        )

        report_gravitational_moments(moments)
    
    # Model writing
    if output_options.model.save:
        j_eq = (num.angular_resolution - 1) // 2
        omega_equator = rot.omega(r2d[:, j_eq], 0.0)

        write_deformed_model(
            output_options.model.filename,
            r2d=r2d,
            additional_variables=additional_variables,
            zeta=zeta,
            p=p,
            rho=rho,
            phi_eff=phi_eff,
            omega_equator=omega_equator,
            mass=mass,
            radius=radius,
            rotation_target=rotation_target,
            G=G,
            dimensional=output_options.model.dimensional,
        )
        
    return result

    
#----------------------------------------------------------------#
#                   Radiative flux computation                   #
#----------------------------------------------------------------#
def compute_radiative_flux(
    r2d,
    zeta,
    num: RadialNumerics,
    options: RadiativeFluxOptions,
):
    """
    Compute the surface radiative-flux distribution from flux lines.

    Characteristics are integrated through the deformed mapping from
    the surface to the chosen inner origin. Flux conservation along
    these lines determines the normalized surface distribution, which
    is returned as Legendre coefficients.
    """
    t_grid = num.t
    L = num.max_degree
    spl_order = num.spline_order

    z0 = options.origin
    j_lines = options.n_lines

    # Initial angular positions of the downward characteristics
    t_flux, weights_flux = roots_legendre(2 * j_lines)
    t_dw = t_flux[:j_lines]
    weights_dw = weights_flux[:j_lines]

    # Restrict the mapping to the radiative-flux domain
    flux_domain = zeta >= z0
    z = zeta[flux_domain]
    r2d_flux = r2d[flux_domain]

    # Integration coordinate measured inward from the surface
    depth = (1.0 - z)[::-1]

    # Metric terms
    der = compute_mapping_derivatives(
        r2d_flux,
        z,
        t_grid,
        max_degree=L,
        spline_order=spl_order,
        domain_ranges=(slice(None),),
    )
    geo = compute_mapping_geometry(
        r2d_flux,
        der,
        t_grid,
    )

    r_l   = pl_project_2D(r2d_flux, L)
    rhs_l = pl_project_2D(geo.gg, L, even=False)
    jac_l = pl_project_2D(geo.jacobian, L)

    # Characteristic equation dt / d(depth)
    def flux_line_rhs(depth_eval, t_eval):
        rhs_t = np.atleast_2d(
            pl_eval_2D(
                rhs_l,
                t_eval.ravel(),
            )
        )

        return np.array([
            interpolate_func(
                depth,
                -rhs_tk[::-1],
                k=3,
            )(depth_eval)
            for rhs_tk in rhs_t.T
        ]).reshape(t_eval.shape)

    def flux_line_jacobian(depth_eval, t_eval):
        jac_t = np.atleast_2d(
            pl_eval_2D(
                jac_l,
                t_eval.ravel(),
            )
        )

        jacobian = np.diag([
            interpolate_func(
                depth,
                -jac_tk[::-1],
                k=3,
            )(depth_eval)
            for jac_tk in jac_t.T
        ])

        return jacobian.reshape(-1, *t_eval.shape)

    # Solve the characteristics from the surface to z0
    start = time.perf_counter()

    solution = solve_ivp(
        fun=flux_line_rhs,
        t_span=(0.0, 1.0 - z0),
        y0=t_dw,
        method="LSODA",
        dense_output=True,
        rtol=1.0e-4,
        atol=1.0e-4,
        jac=flux_line_jacobian,
        vectorized=True,
    )

    t_lines = solution.sol(depth).T[::-1]

    finish = time.perf_counter()
    print(
        "\nFlux lines found in "
        f"{finish - start:.2f} secs"
    )

    # Mapping and angular derivative along the characteristics
    r_lines, r_t_lines = np.moveaxis(
        np.array([
            pl_eval_2D(r_l[i], t_i, der=1)
            for i, t_i in enumerate(t_lines)
        ]),
        0,
        1,
    )

    # Radial derivative and relative divergence along the lines
    r_z_l = pl_project_2D(der.r_z, L)
    divrel_z_l = pl_project_2D(geo.divrelz, L)

    r_z_lines = np.array([
        pl_eval_2D(r_z_l[i], t_i)
        for i, t_i in enumerate(t_lines)
    ])

    divrel_z_lines = np.array([
        pl_eval_2D(divrel_z_l[i], t_i)
        for i, t_i in enumerate(t_lines)
    ])

    # Flux transport along each characteristic
    Q_z = np.exp([
        -integrate(z, divrel_z_i)
        for divrel_z_i in divrel_z_lines.T
    ])

    surface_factor = (
        r_lines[-1] ** 2
        + (1.0 - t_dw**2)
        * r_t_lines[-1] ** 2
    ) / r_z_lines[-1]

    Q0 = 1.0 / np.sum(Q_z * surface_factor * weights_dw)

    gzz_surface_l = pl_project_2D(geo.gzz[-1], L)
    gzz_surface = pl_eval_2D(gzz_surface_l, t_lines[-1])

    Q_dw = Q0 * Q_z * np.sqrt(np.abs(gzz_surface))

    # Restore the symmetric upper hemisphere
    Q_l = pl_project_2D(np.hstack((Q_dw, Q_dw[::-1])), 2 * j_lines)

    plot_3D_surface(
        r_l[-1],
        Q_l,
        show_T_eff=options.show_effective_temperature,
        res=options.resolution,
        cmap=options.cmap,
    )

    if options.plot_lines:
        r_lines = np.hstack((r_lines,  r_lines[:, ::-1]))
        t_lines = np.hstack((t_lines, -t_lines[:, ::-1]))

        fig, ax = plot_flux_lines(r_lines, t_lines, color="grey")
    else:
        fig, ax = None, None

    return Q_l, (fig, ax)