import time
import numpy        as np
import scipy.sparse as sps
from dataclasses         import dataclass
from numpy.typing        import NDArray
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbtrf, dgbtrs

from ..config            import (
    RotationConfig, 
    SolverOptions,
)
from ..legendre          import (
    find_r_eq, 
    find_r_pol, 
    pl_eval_2D, 
    pl_project_2D,
)
from ..interpolation     import interpolate_func
from ..lagrange          import lagrange_matrix_P
from ..quadrature        import integrate_axisymmetric
from ..models            import (
    Model1D,
    Model2D,
)
from ..mapping           import (
    initialize_mapping, 
    valid_reciprocal_domain,
    compute_mapping_derivatives,
)
from ..hydrostatics      import integrate_pressure
from ..rotation          import (
    RotationState,
    initialize_rotation_state,
)
from .convergence        import ConvergenceTracker
from ..results           import SolverInfo


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
    rho2d = np.zeros_like(r2d)

    for j in j_dw:
        inside = r < r2d[-1, j]
        rho2d[inside, j] = interpolate_func(x=r2d[:, j], y=log_rho, k=spl_order)(r[inside])
        rho2d[inside, j] = np.exp(rho2d[inside, j]) - safety_constant

    rho2d[:, j_up] = rho2d[:, j_dw]

    return pl_project_2D(rho2d, L)
    
    
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
    phi_g_r_l = np.zeros((I, L))
    
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
    phi_g_r_l[:, ::2] = x[0::2] * (2*r1d)  # <- The equation is solved on r^2
    
    if phi_eff is None :
        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        phi_eff_r = pl_eval_2D(phi_g_r_l, 0.0)        
        return phi_g_l, phi_g_r_l, phi_eff, phi_eff_r, poisson_factors
    
    # The effective potential is known up to an additive constant 
    phi_offset = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += phi_offset
    
    return phi_g_l, phi_g_r_l, phi_eff


def evaluate_gravitational_fields(
    r2d,
    phi_g_l,
    phi_g_r_l,
    num: RadialNumerics,
):
    """
    Evaluate the spherical Poisson solution on the material mapping.

    Harmonic fields are first evaluated on the fixed spherical grid and
    then interpolated radially onto r2d(zeta, t).
    """
    t = num.t
    r = num.r1d

    phi_g_grid = pl_eval_2D(phi_g_l, t)
    phi_g_r_grid = pl_eval_2D(phi_g_r_l, t)

    splines = [
        CubicHermiteSpline(
            x=r,
            y=phi_g_grid[:, j],
            dydx=phi_g_r_grid[:, j],
        )
        for j in range(t.size)
    ]

    phi_g = np.array([
        spline(r2d[:, j])
        for j, spline in enumerate(splines)
    ]).T

    phi_g_r = np.array([
        spline(r2d[:, j], nu=1)
        for j, spline in enumerate(splines)
    ]).T

    return phi_g, phi_g_r


def update_mapping(
    phi_g_l,
    phi_g_r_l,
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
    phi_g_int   = pl_eval_2D(phi_g_l,   t[j_dw])
    phi_g_r_int = pl_eval_2D(phi_g_r_l, t[j_dw])

    # Exterior gravitational potential
    l = np.arange(L)
    outside = 1.3
    r_ext = np.linspace(1.0, outside, 101)[1:]

    phi_g_l_ext  = phi_g_l[-1] * r_ext[:, None] ** -(l + 1)
    phi_g_r_l_ext = -(l + 1) * phi_g_l_ext / r_ext[:, None]

    phi_g_ext   = pl_eval_2D(phi_g_l_ext,   t[j_dw])
    phi_g_r_ext = pl_eval_2D(phi_g_r_l_ext, t[j_dw])

    # Full radial domain
    r_tot = np.hstack((r, r_ext))
    phi_g   = np.vstack((phi_g_int, phi_g_ext))
    phi_g_r = np.vstack((phi_g_r_int, phi_g_r_ext))

    # Find a rotation rate consistent with the equatorial radius
    safe = r_tot > 0.0
    r_safe = r_tot[safe]

    phi_g_eq = phi_g[safe, j_eq]
    phi_g_eq_r = phi_g_r[safe, j_eq]
    phi_c_eq, phi_c_eq_r = rot.phi_c(r_safe, 0.0) / r_safe**3
    phi_c_eq_r -= 3.0 * phi_c_eq / r_safe

    phi_eq   = phi_g_eq   + phi_c_eq
    phi_eq_r = phi_g_eq_r + phi_c_eq_r

    r_eq_new = CubicHermiteSpline(x=phi_eq, y=r_safe, dydx=phi_eq_r**-1)(phi_eff[-1])
    omega_eq_new = rot.omega_eq * r_eq_new**-1.5
    rot_new = rot.with_omega_eq(omega_eq_new)

    # Centrifugal potential
    phi_c, phi_c_r = rot_new.phi_c2d_with_derivative(r_tot[:, None], t[j_dw])

    # Total potential
    phi   = phi_g   + phi_c
    phi_r = phi_g_r + phi_c_r

    valid = valid_reciprocal_domain(r_tot, phi_r)

    # Refined central domain
    lim = 1.0e-1
    lim_idx = np.max(np.argwhere(r_tot < lim)) + 1

    r_cnt = lim * np.linspace(0.0, 1.0, 5 * lim_idx) ** 2

    phi_g_cnt = np.array([
        CubicHermiteSpline(
            x=r,
            y=phi_g_int[:, j],
            dydx=phi_g_r_int[:, j],
        )(r_cnt)
        for j in j_dw
    ]).T

    phi_c_cnt = np.array([
        rot_new.phi_c(r_cnt, t_j)[0]
        for t_j in t[j_dw]
    ]).T

    phi_cnt = phi_g_cnt + phi_c_cnt

    # Estimate the radius at each target equipotential
    r2d_dw_origin = np.zeros_like(j_dw)
    r2d_dw_center = np.array([
        interpolate_func(x=pk, y=r_cnt, k=k)(phi_eff[1:lim_idx])
        for pk in phi_cnt.T
    ]).T
    r2d_dw_envelope = np.array([
        interpolate_func(x=pk[vk], y=r_tot[vk], k=k)(phi_eff[lim_idx:])
        for pk, vk in zip(phi.T, valid.T)
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
) -> tuple[Model2D, None, SolverInfo]:
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
    phi_g_l, phi_g_r_l, phi_eff, phi_eff_r, poisson_factors = solve_gravitational_potential(r2d, rho, num)
    
    # Find pressure
    p = integrate_pressure(
        zeta,
        rho,
        phi_eff_r,
        surface_pressure,
        spline_order=num.spline_order,
    )
    
    # Iterative centrifugal deformation
    conv = ConvergenceTracker.start(
        find_r_pol(r2d, L),
        solver_name="Radial",
        quantity_name="polar_radius",
        tolerance=mapping_precision,
        max_iterations=max_iterations,
        verbose=solver_options.verbose,
    )
    
    while not conv.converged:
        conv.check_iteration_limit()

        # Current rotation rate
        rotation_cap = (conv.iterations + 1) / full_rate * rotation_target
        rot = rot.with_omega_eq(min(rotation_target, rotation_cap))

        # Effective potential computation
        phi_g_l, phi_g_r_l, phi_eff = (
            solve_gravitational_potential(
                r2d,
                rho,
                num,
                phi_eff=phi_eff,
                poisson_factors=poisson_factors,
            )
        )

        # Find a new estimate for the mapping
        r2d, rot = update_mapping(
            phi_g_l,
            phi_g_r_l,
            phi_eff,
            num,
            rot,
        )

        # Renormalisation
        r_corr = find_r_eq(r2d, L)
        m_corr = integrate_axisymmetric(r2d, rho)

        radius *= r_corr
        mass   *= m_corr

        r2d       /= r_corr
        rho       /= m_corr    / r_corr**3
        phi_eff   /= m_corr    / r_corr
        p         /= m_corr**2 / r_corr**4

        # Update convergence
        conv.update(find_r_pol(r2d, L))
        
    
    # Compute final 2D potentials
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=spl_order,
        domain_ranges=(slice(None),),
    )

    phi_g, phi_g_r = evaluate_gravitational_fields(
        r2d,
        phi_g_l,
        phi_g_r_l,
        num,
    )
    phi_g_z = phi_g_r * der.r_z

    phi_c, phi_c_r = rot.phi_c2d_with_derivative(r2d, t)
    phi_c_z = phi_c_r * der.r_z

    phi_eff_z = interpolate_func(
        zeta,
        phi_eff,
        der=1,
        k=spl_order,
    )(zeta)

    omega = rot.omega2d(r2d, t)
    
    # 2D Model
    model2d = Model2D(
        G=G,
        surface_pressure=float(p[-1]),
        mass=mass,
        radius=radius,
        omega_eq=rot.omega_eq,

        zeta=zeta.copy(),
        t=t.copy(),
        r2d=r2d.copy(),

        rho=rho.copy(),
        p=p.copy(),
        additional_variables=tuple(
            var.copy() for var in additional_variables
        ),

        phi_eff=phi_eff.copy(),
        phi_eff_z=phi_eff_z.copy(),

        phi_g=phi_g.copy(),
        phi_g_z=phi_g_z.copy(),

        phi_c=phi_c.copy(),
        phi_c_z=phi_c_z.copy(),

        omega=omega.copy(),
        domains=model.domains,
    )

    # Solver info
    elapsed_time = time.perf_counter() - start
    conv.report_convergence(elapsed_time=elapsed_time)
    
    info = SolverInfo(
        method="radial",
        iterations=conv.iterations,
        tolerance=conv.tolerance,
        error=conv.error,
        polar_radius_history=np.asarray(conv.history),
        rotation_target=rotation_target,
        elapsed_time=elapsed_time,
    )
        
    return model2d, None, info