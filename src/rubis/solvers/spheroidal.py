import time
import numpy             as np
import scipy.sparse      as sps
import scipy.special     as sp
from dataclasses         import dataclass
from gc                  import collect
from numpy.typing        import NDArray
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbsv

from ..legendre          import (
    find_r_eq, 
    find_r_pol, 
    pl_eval_2D, 
    pl_project_2D,
)
from ..interpolation     import interpolate_func
from ..lagrange          import build_lagrange_operators
from ..quadrature        import (
    integrate,
    integrate_axisymmetric,
)
from ..config            import (
    RotationConfig, 
    SolverOptions,
)
from ..models            import (
    Model1D,
    Model2D,
    VacuumModel2D,
)
from ..domains           import (
    DomainLayout,
    find_domains,
)
from ..mapping           import (
    initialize_mapping, 
    valid_reciprocal_domain,
    compute_mapping_derivatives,
    extend_mapping,
)
from ..hydrostatics      import integrate_pressure
from ..poisson           import compute_poisson_couplings
from ..rotation          import (
    RotationState,
    initialize_rotation_state,
)
from .convergence        import ConvergenceTracker
from ..results           import SolverInfo


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True, kw_only=True)
class SpheroidalNumerics:
    """
    Fixed numerical representation used by the spheroidal solver.

    It stores the full multidomain zeta coordinate, the angular
    collocation grid, and the domainwise operators used by the
    spheroidal Poisson problem.
    """

    zeta: FloatArray
    t: FloatArray
    domains: DomainLayout

    max_degree: int
    spline_order: int
    lagrange_order: int

    Lsp: tuple[sps.spmatrix, ...]
    Dsp: tuple[sps.spmatrix, ...]

    @property
    def n_points(self) -> int:
        """Total number of radial points, including the exterior."""
        return self.zeta.size

    @property
    def vacuum_domain_id(self) -> int:
        """Index of the exterior vacuum domain."""
        return self.domains.n_domains - 1

    @property
    def n_material_points(self) -> int:
        """Number of points in the material domains."""
        return int(
            self.domains.domain_edges[-2]
        )

    @property
    def n_vacuum_points(self) -> int:
        """Number of points in the vacuum domain."""
        return (
            self.n_points
            - self.n_material_points
        )

    @property
    def material_slice(self) -> slice:
        """Contiguous slice containing all material domains."""
        return slice(
            0,
            self.n_material_points,
        )

    @property
    def vacuum_slice(self) -> slice:
        """Contiguous slice containing the vacuum domain."""
        return slice(
            self.n_material_points,
            None,
        )

    @property
    def material_ranges(
        self,
    ) -> tuple[range, ...]:
        """Ranges of the material domains."""
        return self.domains.domain_ranges[:-1]

    @property
    def vacuum_range(self) -> range:
        """Range of the exterior vacuum domain."""
        return self.domains.domain_ranges[-1]

    @property
    def material_interface_end_indices(
        self,
    ) -> IntArray:
        """Lower copies of interfaces between material domains."""
        return (
            self.domains
            .interface_end_indices[:-1]
        )

    @property
    def material_interface_start_indices(
        self,
    ) -> IntArray:
        """Upper copies of interfaces between material domains."""
        return (
            self.domains
            .interface_start_indices[:-1]
        )

    @property
    def angular_resolution(self) -> int:
        return self.t.size


def initialize_spheroidal_numerics(
    r1d,
    t,
    options: SolverOptions,
) -> SpheroidalNumerics:
    """
    Build the fixed multidomain representation of the spheroidal solver.

    The material coordinate is extended by one vacuum domain, and the
    interpolation and differentiation operators are constructed
    independently in each domain.
    """
    I_ext = options.external_domain_res

    x_ext = np.linspace(1.0, 2.0, I_ext)
    zeta_ext = 1.0 + sp.betainc(2, 2, x_ext - 1.0)

    zeta = np.hstack((r1d, zeta_ext))

    domains = find_domains(zeta)

    Lsp = []
    Dsp = []

    for i_dom in domains.domain_ids:
        idx = domains.domain_ranges[i_dom]

        lag_mat = build_lagrange_operators(
            zeta[idx],
            order=options.lagrange_order,
        )

        Lsp.append(
            sps.dia_matrix(lag_mat[..., 0])
        )
        Dsp.append(
            sps.dia_matrix(lag_mat[..., 1])
        )

    return SpheroidalNumerics(
        zeta=zeta,
        t=t,
        domains=domains,
        max_degree=options.max_degree,
        spline_order=options.spline_order,
        lagrange_order=options.lagrange_order,
        Lsp=tuple(Lsp),
        Dsp=tuple(Dsp),
    )



def assemble_poisson_system(
    cpl,
    rhs_l,
    num: SpheroidalNumerics,
    rescale=True,
):
    """
    Assemble the multidomain band system for Poisson's equation.

    The radial operators and harmonic couplings are combined in every
    domain, with regularity, interface, and vacuum boundary conditions.
    """
    I = num.n_points
    L = num.max_degree
    lag_order = num.lagrange_order
    domains = num.domains

    L_even = (L + 1) // 2
    block_size = 2 * L_even

    kl = ku = (2 * lag_order + 1) * L_even - 1
    system_size = block_size * I
    coefficient_rows = 2 * lag_order * block_size

    degrees = 2 * np.arange(L_even)
    identity = np.eye(L_even)

    b = np.zeros(system_size)
    coefs = np.empty((coefficient_rows, system_size))

    for d, idx in enumerate(domains.domain_ranges):
        start = domains.domain_edges[d]
        stop = domains.domain_edges[d + 1]
        size = domains.domain_sizes[d]

        beg_i = (2 * start + 1) * L_even
        end_i = (2 * stop - 1) * L_even
        beg_j = (2 * start + 0) * L_even
        end_j = (2 * stop - 0) * L_even

        Lsp = num.Lsp[d]
        Dsp = num.Dsp[d]

        Lsp_broad = Lsp.data[::-1, :, None, None]
        Dsp_broad = Dsp.data[::-1, :, None, None]

        # Matter source in material domains
        if d != num.vacuum_domain_id:
            b[beg_i:end_i:2] = (Lsp @ rhs_l[idx]).ravel()

        # Local differential operator
        temp = np.empty(
            (2 * lag_order, size, block_size, block_size)
        )

        temp[..., 0::2, 0::2] = (
            +Dsp_broad * cpl.zz[idx]
            - Lsp_broad * cpl.zt[idx]
        )
        temp[..., 0::2, 1::2] = -Lsp_broad * cpl.tt[idx]
        temp[..., 1::2, 0::2] = +Lsp_broad * identity
        temp[..., 1::2, 1::2] = -Dsp_broad * identity

        coefs[:, beg_j:end_j] = np.moveaxis(
            temp, 2, 1
        ).reshape(coefficient_rows, block_size * size)

        del temp
        collect()

        # Central regularity or inner interface conditions
        if d == 0:
            coefs[
                ku - 2 * L_even + 1 : ku - L_even + 1,
                0 : block_size : 2,
            ] = np.diag((1,) + (0,) * (L_even - 1))

            coefs[
                ku - 2 * L_even + 1 : ku - L_even + 1,
                1 : block_size : 2,
            ] = np.diag((0,) + (1,) * (L_even - 1))

        else:
            coefs[
                ku - 3 * L_even + 1 + 0 - 0 :
                ku - L_even + 1 : 2,
                beg_j + 0 : beg_j + block_size : 2,
            ] = -cpl.boundary[idx[0]]

            coefs[
                ku - 3 * L_even + 1 + 1 - 0 :
                ku - L_even + 1 : 2,
                beg_j + 1 : beg_j + block_size : 2,
            ] = -identity

        # Vacuum decay or outer interface conditions
        if d == num.vacuum_domain_id:
            coefs[
                ku - L_even + 1 : ku + 1,
                -block_size + 0 :: 2,
            ] = identity

            coefs[
                ku - L_even + 1 : ku + 1,
                -block_size + 1 :: 2,
            ] = np.diag((degrees + 1) / 2)

        else:
            coefs[
                ku - L_even + 1 + 0 - 0 :
                ku + L_even + 1 : 2,
                end_j - block_size + 0 : end_j : 2,
            ] = cpl.boundary[idx[-1]]

            coefs[
                ku - L_even + 1 + 1 - 0 :
                ku + L_even + 1 : 2,
                end_j - block_size + 1 : end_j : 2,
            ] = identity

    col_scale = np.ones(system_size)

    if rescale:
        # Find the row scaling factors
        row_max = np.abs(
            coefs.reshape(coefficient_rows, I, block_size)
        ).max(axis=2)

        offset = (2 * lag_order - 1) * L_even
        row_scale = np.zeros(system_size + 2 * offset)

        for i in range(I):
            i0 = block_size * i
            row_scale[i0 : i0 + coefficient_rows] = np.maximum(
                row_scale[i0 : i0 + coefficient_rows],
                row_max[:, i],
            )

        row_scale = np.divide(
            1.0,
            row_scale,
            out=np.zeros_like(row_scale),
            where=row_scale != 0.0,
        )

        # Apply the row scaling factors
        b *= row_scale[offset:-offset]

        row_scale2d = np.array([
            row_scale[
                block_size * i :
                block_size * i + coefficient_rows
            ]
            for i in range(I)
        ]).T

        coefs = (
            coefs.reshape(coefficient_rows, I, block_size)
            * row_scale2d[..., None]
        ).reshape(coefs.shape)

        # Find and apply the column scaling factors
        col_scale = 1.0 / np.abs(coefs).max(axis=0)
        coefs *= col_scale

    # Convert the coefficients to LAPACK band storage
    ab = np.zeros((2 * kl + ku + 1, system_size))
    mask = np.zeros(
        (system_size, kl + ku + 1),
        dtype=bool,
    )

    for q in range(block_size):
        mask[
            q::block_size,
            L - q : kl + ku + 1 - q,
        ] = True

    ab[kl:].T[mask] = coefs.T.ravel()

    return ab, b, col_scale, kl, ku


def solve_gravitational_potential(
    r2d,
    rho,
    num: SpheroidalNumerics,
    phi_eff=None,
    *,
    rescale=True,
):
    """
    Solve Poisson's equation in the multidomain spheroidal coordinates.

    The matter source is projected on the internal mapping, while the
    differential operator is assembled on the mapping extended through
    the exterior vacuum domain.
    """
    I = num.n_points
    L = num.max_degree
    t = num.t
    zeta = num.zeta
    domains = num.domains

    L_even = (L + 1) // 2

    # Geometry of the internal material mapping
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=num.spline_order,
        domain_ranges=num.material_ranges,
    )

    # Matter source on the internal domains
    r2rz_l = pl_project_2D(
        r2d**2 * der.r_z,
        L,
    ) / (np.arange(L) + 0.5)

    rhs_l = 4 * np.pi * rho[:, None] * r2rz_l[:, ::2]

    # Extend the mapping into the exterior vacuum domain
    zeta_ext = zeta[num.vacuum_slice]

    r2d_ext, der_ext = extend_mapping(
        r2d,
        der,
        zeta_ext,
    )

    cpl = compute_poisson_couplings(
        r2d_ext,
        der_ext,
        t,
        max_degree=L,
        alpha=2,
    )

    ab, b, col_scale, kl, ku = (
        assemble_poisson_system(
            cpl,
            rhs_l,
            num,
            rescale=rescale,
        )
    )

    # Solve the complete coupled band system
    *_, x, info = dgbsv(kl, ku, ab, b)

    del ab, b
    collect()

    if info != 0:
        raise ValueError(
            "Unable to solve the spheroidal Poisson system. "
            f"LAPACK returned info={info}."
        )

    x *= col_scale

    # Recover the even Legendre harmonics
    phi_g_l   = np.zeros((I, L))
    phi_g_z_l = np.zeros((I, L))

    phi_g_l[:, ::2]   = x[1::2].reshape(I, L_even)
    phi_g_z_l[:, ::2] = x[0::2].reshape(I, L_even)

    if phi_eff is None:
        phi_eff   = pl_eval_2D( phi_g_l, 0.0)
        phi_eff_z = pl_eval_2D(phi_g_z_l, 0.0)

        return (
            phi_g_l,
            phi_g_z_l,
            phi_eff,
            phi_eff_z,
        )

    phi_offset = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += phi_offset

    return phi_g_l, phi_g_z_l, phi_eff


def update_mapping(
    r2d,
    phi_g_l,
    phi_g_z_l,
    phi_eff,
    phi_eff_z,
    num: SpheroidalNumerics,
    rot: RotationState,
):
    """
    Construct the next material mapping from the total potential.

    The current mapping and gravitational solution are interpolated on
    an adaptive full-domain coordinate. Reciprocal interpolation at fixed
    effective potential then recovers the internal mapping r(zeta, t).
    """
    I = num.n_material_points
    J = num.angular_resolution
    L = num.max_degree

    zeta = num.zeta
    t = num.t
    domains = num.domains
    spl_order = num.spline_order

    targets = phi_eff[num.material_slice].copy()

    # Extend the current mapping through the vacuum domain
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=spl_order,
        domain_ranges=num.material_ranges,
    )
    zeta_ext = zeta[num.vacuum_slice]
    r2d_ext, der_ext = extend_mapping(r2d, der, zeta_ext)

    # Lower angular half-domain, including the equator
    j_eq = (J - 1) // 2
    j_dw = np.arange(j_eq + 1)

    phi_g   = pl_eval_2D(phi_g_l,   t[j_dw])
    phi_g_z = pl_eval_2D(phi_g_z_l, t[j_dw])
    phi_g_r = phi_g_z / der_ext.r_z[:, j_dw]

    # Correct the equatorial rotation rate
    safe = zeta > 0.5
    r_safe = r2d_ext[safe, j_eq]

    phi_g_eq = phi_g[safe, j_eq]
    phi_g_eq_r = phi_g_r[safe, j_eq]
    phi_c_eq, phi_c_eq_r = rot.phi_c(r_safe, 0.0) / r_safe**3
    phi_c_eq_r -= 3.0 * phi_c_eq / r_safe

    phi_eq   = phi_g_eq   + phi_c_eq
    phi_eq_r = phi_g_eq_r + phi_c_eq_r

    unique_r = find_domains(r_safe).unique_indices
    r_eq_new = CubicHermiteSpline(
        x=phi_eq[unique_r],
        y=r_safe[unique_r],
        dydx=phi_eq_r[unique_r]**-1,
    )(targets[-1])

    rot_new = rot.with_omega_eq(
        rot.omega_eq * r_eq_new**-1.5
    )

    # Build an adaptive coordinate for reciprocal interpolation
    new_res = int(1.0 / np.finfo(float).eps**0.2)

    d2phi_eff = np.hstack((
        0.0,
        np.abs(np.diff(phi_eff_z[domains.unique_indices])),
    ))

    zeta_new = interpolate_func(
        d2phi_eff.cumsum(),
        zeta[domains.unique_indices],
        k=1,
    )(np.linspace(0.0, d2phi_eff.sum(), new_res))

    zeta_new = 2.0 * (
        zeta_new - zeta_new[0]
    ) / (
        zeta_new[-1] - zeta_new[0]
    )

    # Interpolate the mapping and gravitational potential
    r_splines = [
        CubicHermiteSpline(
            x=zeta[domains.unique_indices],
            y=r2d_ext[domains.unique_indices, j],
            dydx=der_ext.r_z[domains.unique_indices, j],
        )
        for j in j_dw
    ]
    phi_splines = [
        CubicHermiteSpline(
            x=zeta[domains.unique_indices],
            y=phi_g[domains.unique_indices, j],
            dydx=phi_g_z[domains.unique_indices, j],
        )
        for j in j_dw
    ]

    r_itp, r_z_itp = [
        np.array([
            spline(zeta_new, nu=nu)
            for spline in r_splines
        ]).T
        for nu in (0, 1)
    ]
    
    # Gravitational potential
    phi_g_itp, phi_g_itp_z = [
        np.array([
            spline(zeta_new, nu=nu)
            for spline in phi_splines
        ]).T
        for nu in (0, 1)
    ]
    phi_g_itp_r = phi_g_itp_z / r_z_itp

    # Centrifugal potential
    phi_c_itp, phi_c_itp_r = rot_new.phi_c2d_with_derivative(r_itp, t[j_dw])

    # Total potential
    phi_itp   = phi_g_itp   + phi_c_itp
    phi_itp_r = phi_g_itp_r + phi_c_itp_r

    # Invert the total potential at the target values
    valid = valid_reciprocal_domain(zeta_new, phi_itp_r)

    r2d_dw = np.zeros_like(r2d[:, j_dw])
    r2d_dw[1:] = np.array([
        CubicHermiteSpline(
            x=phi_j[valid_j],
            y=r_j[valid_j],
            dydx=phi_j_r[valid_j]**-1,
        )(targets[1:])
        for r_j, phi_j, phi_j_r, valid_j in zip(
            r_itp.T,
            phi_itp.T,
            phi_itp_r.T,
            valid.T,
        )
    ]).T

    # Restore duplicated internal interfaces
    r2d_dw[num.material_interface_start_indices] = (
        r2d_dw[num.material_interface_end_indices]
    )
    
    r2d_up = np.flip(r2d_dw, axis=1)[:, 1:]
    r2d_new = np.hstack((r2d_dw, r2d_up))

    return r2d_new, rot_new


def solve_spheroidal(
    model: Model1D,
    rotation_config: RotationConfig,
    solver_options: SolverOptions,
) -> tuple[Model2D, VacuumModel2D, SolverInfo]:
    """
    Compute rotational deformation on a multidomain spheroidal grid.

    Poisson's equation is solved directly in the evolving material
    coordinates, extended by an exterior vacuum domain. The mapping,
    rotation state, mass, and radius are iterated until convergence.
    """
    start = time.perf_counter()

    # Physical model
    G = model.G
    surface_pressure = model.surface_pressure
    mass = model.mass
    radius = model.radius

    r1d = model.r.copy()
    rho = model.rho.copy()
    additional_variables = model.additional_variables

    # Angular grid and initial material mapping
    r2d, t = initialize_mapping(
        r1d,
        solver_options.angular_resolution,
    )

    # Fixed multidomain numerical representation
    num = initialize_spheroidal_numerics(
        r1d,
        t,
        solver_options,
    )

    # Modern local dimension names
    L = num.max_degree
    spl_order = num.spline_order

    zeta = num.zeta
    domains = num.domains

    rot = initialize_rotation_state(rotation_config)
    rotation_target = rotation_config.target

    rescale_ab = solver_options.rescale_ab
    full_rate = solver_options.full_rate
    mapping_precision = solver_options.mapping_precision
    max_iterations = solver_options.max_iterations
    
    # Initialisation for the effective potential
    phi_g_l, phi_g_z_l, phi_eff, phi_eff_z = (
        solve_gravitational_potential(
            r2d,
            rho,
            num,
            rescale=rescale_ab,
        )
    )
    
    # Find pressure
    material = num.material_slice

    p = integrate_pressure(
        num.zeta[material],
        rho,
        phi_eff_z[material],
        surface_pressure,
        unique_indices=model.domains.unique_indices,
        spline_order=num.spline_order,
    )
    
    # Iterative centrifugal deformation
    conv = ConvergenceTracker.start(
        find_r_pol(r2d, L),
        solver_name="Spheroidal",
        quantity_name="R_pol",
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
        phi_g_l, phi_g_z_l, phi_eff = (
            solve_gravitational_potential(
                r2d,
                rho,
                num,
                phi_eff=phi_eff,
                rescale=rescale_ab,
            )
        )

        # Find a new estimate for the mapping
        r2d, rot = update_mapping(
            r2d,
            phi_g_l,
            phi_g_z_l,
            phi_eff,
            phi_eff_z,
            num,
            rot,
        )

        # Renormalisation
        r_corr = find_r_eq(r2d, L)
        m_corr = integrate_axisymmetric(r2d, rho, domains=num.material_ranges)

        radius *= r_corr
        mass   *= m_corr

        r2d       /= r_corr
        rho       /= m_corr    / r_corr**3
        phi_eff   /= m_corr    / r_corr
        phi_eff_z /= m_corr    / r_corr
        p         /= m_corr**2 / r_corr**4

        # Update convergence
        conv.update(find_r_pol(r2d, L))
        
    
    # Extend the converged mapping through the vacuum domain
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=spl_order,
        domain_ranges=num.material_ranges,
    )
    z_ext = zeta[num.vacuum_slice]
    r2d_ext, der_ext = extend_mapping(r2d, der, z_ext)
    
    # Compute final 2D potentials
    phi_g = pl_eval_2D(phi_g_l, t)
    phi_g_z = pl_eval_2D(phi_g_z_l, t)

    phi_c, phi_c_r = rot.phi_c2d_with_derivative(
        r2d_ext,
        t,
    )
    phi_c_z = phi_c_r * der_ext.r_z

    omega = rot.omega2d(r2d_ext, t)
    
    phi_eff_vac = phi_g + phi_c
    phi_eff_z_vac = phi_g_z + phi_c_z
    
    material = num.material_slice
    vacuum_domain = num.vacuum_slice
    
    # 2D Model
    model2d = Model2D(
        G=G,
        surface_pressure=float(p[-1]),
        mass=mass,
        radius=radius,
        omega_eq=rot.omega_eq,

        zeta=zeta[material].copy(),
        t=t.copy(),
        r2d=r2d.copy(),

        rho=rho.copy(),
        p=p.copy(),
        additional_variables=tuple(
            var.copy()
            for var in additional_variables
        ),

        phi_eff=phi_eff[material].copy(),
        phi_eff_z=phi_eff_z[material].copy(),

        phi_g=phi_g[material].copy(),
        phi_g_z=phi_g_z[material].copy(),

        phi_c=phi_c[material].copy(),
        phi_c_z=phi_c_z[material].copy(),

        omega=omega[material].copy(),
        domains=model.domains,
    )
    
    # Vacuum Model
    vacuum = VacuumModel2D(
        G=G,
        mass=mass,
        radius=radius,
        omega_eq=rot.omega_eq,

        zeta=zeta[vacuum_domain].copy(),
        t=t.copy(),
        r2d=r2d_ext[vacuum_domain].copy(),

        phi_g=phi_g[vacuum_domain].copy(),
        phi_g_z=phi_g_z[vacuum_domain].copy(),

        phi_c=phi_c[vacuum_domain].copy(),
        phi_c_z=phi_c_z[vacuum_domain].copy(),

        phi_eff=(
            phi_eff_vac[vacuum_domain]
            .copy()
        ),
        phi_eff_z=(
            phi_eff_z_vac[vacuum_domain]
            .copy()
        ),

        omega=omega[vacuum_domain].copy(),

        domains=find_domains(
            zeta[vacuum_domain]
        ),
    )
    
    # Solver Info
    elapsed_time = time.perf_counter() - start
    conv.report_convergence(elapsed_time=elapsed_time)
    
    info = SolverInfo(
        method="spheroidal",
        iterations=conv.iterations,
        tolerance=conv.tolerance,
        error=conv.error,
        polar_radius_history=np.asarray(conv.history),
        rotation_target=rotation_target,
        elapsed_time=elapsed_time,
    )
    
    return model2d, vacuum, info
        
        
        

    
    
    
    

    