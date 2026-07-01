import time
import numpy             as np
import scipy.sparse      as sps
import scipy.special     as sp
from dataclasses         import dataclass
from gc                  import collect
from numpy.typing        import NDArray
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbsv
from scipy.special       import roots_legendre, eval_legendre

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
from ..config import (
    OutputOptions,
    RotationConfig, 
    SolverOptions,
)
from ..models            import Model1D
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
from ..poisson           import compute_poisson_couplings
from ..rotation_profiles import configure_rotation_profile
from ..results           import SpheroidalResult
from ..io.legacy         import write_model
from ..plotting          import (
    phi_g_harmonics,
    plot_f_map,
)


FloatArray = NDArray[np.float64]


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
    def n_internal_points(self) -> int:
        return int(np.count_nonzero(self.domains.internal_mask))

    @property
    def n_external_points(self) -> int:
        return int(np.count_nonzero(self.domains.external_mask))

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

        lag_mat = lagrange_matrix_P(
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


def find_gravitational_moments(r2d, t, rho, max_degree=14) :
    """
    Find the gravitational moments up to max_degree.

    Parameters
    ----------
    mapping : array_like, shape (N, M)
        Isopotential mapping.
    t : array_like, shape (M, )
        Value of cos(theta).
    rho : array_like, shape (N, )
        Density profile (the same in each direction).
    max_degree : int, optional
        Maximum degree for the gravitational moments. The default is 14.
        
    Returns
    -------
    None.

    """
    print(
        "\n+-----------------------+",
        "\n| Gravitational moments |", 
        "\n+-----------------------+\n"
    )
    for l in range(0, max_degree+1, 2):
        m_l = integrate2D(
            r2d, rho[:, None] * r2d ** l * eval_legendre(l, t), 
            domains=domains.domain_ranges[:-1], k=KSPL
        )
        print("Moment n°{:2d} : {:+.10e}".format(l, m_l))


def find_pressure(rho, dphi_eff, P0) :
    """
    Find the pressure evaluated on zeta thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : array_like, shape (N, )
        Density profile.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative with respect to zeta.
    P0 : float
        Surface pressure.

    Returns
    -------
    P : array_like, shape (N, )
        Pressure profile.

    """
    dP = - rho * dphi_eff[domains.internal_mask]        
    P  = interpolate_func(
        1 - zeta[domains.unique_internal_indices][::-1], 
        -dP[domains.unique_internal_indices][::-1], 
        der=-1, 
        k=KSPL, 
        prim_cond=(0, P0),
    )(1-zeta[domains.internal_mask][::-1])[::-1]
    
    return P


def find_Poisson_coefs(kl, ku, cpl, rhs_l, rescale) :
    """
    Finds the coefficients to fill the Poisson matrix (thanks
    to cpl) and the right-hand side (using rhs).

    Parameters
    ----------
    ku : integer
        Number of upper band in the matrix.
    kl : integer
        Number of lower band in the matrix.
    cpl : PoissonCouplings
        Harmonic couplings of the Poisson operator.
    rhs_l : array_like, shape (N, Nl)
        right-hand side of Poisson's equation when projected
        onto the Legendre polynomials.
    rescale : boolean
        Whether to rescale the Poisson's matrix coefficients
        before performing the LU decomposition.

    Returns
    -------
    coefs : array_like, shape (kl+ku-L+1, 2*(N+NE)*Nl)
        Coeffcients in the Poisson matrix.
    b : array_like, shape (2*(N+NE)*Nl, )
        Right-hand side of the linear system.
    """   
    # Initialisation
    Nl = (L+1)//2
    Mb = 2*Nl*(N+NE)
    dj = 2*Nl*2*KLAG
    l  = 2*np.arange(Nl)
    b     = np.zeros((Mb, ))
    coefs = np.empty((kl+ku-L+1, Mb))
    
    for d, D in enumerate(domains.domain_ranges) : 
        
        # Domain properties
        beg_i, end_i = (2*domains.domain_edges[d]+1)*Nl, (2*domains.domain_edges[d+1]-1)*Nl
        beg_j, end_j = (2*domains.domain_edges[d]+0)*Nl, (2*domains.domain_edges[d+1]-0)*Nl
        Lsp_d_broad = Lsp[d].data[::-1, :, None, None]
        Dsp_d_broad = Dsp[d].data[::-1, :, None, None]
        size = domains.domain_sizes[d]
        
        # Vector filling
        if d < domains.n_domains - 1 : 
            b[beg_i:end_i:2] = (Lsp[d] @ rhs_l[D]).flatten()
            
        # Main matrix parts filling
        temp = np.empty((2*KLAG, size, 2*Nl, 2*Nl))
        temp[..., 0::2, 0::2] = (
            + Dsp_d_broad * cpl.zz[D]
            - Lsp_d_broad * cpl.zt[D]
        )
        temp[..., 0::2, 1::2] = - Lsp_d_broad * cpl.tt[D]
        temp[..., 1::2, 0::2] = + Lsp_d_broad * np.eye(Nl)
        temp[..., 1::2, 1::2] = - Dsp_d_broad * np.eye(Nl)
        coefs[:, beg_j:end_j] = np.moveaxis(temp, 2, 1).reshape(
            (2*KLAG*2*Nl, 2*size*Nl)
        )
        del temp; collect()
            
        # Inner boundary conditions 
        if d == 0 :
            coefs[ku-2*Nl+1:ku-Nl+1, 0:2*Nl:2] = np.diag((1, ) + (0, )*(Nl-1))
            coefs[ku-2*Nl+1:ku-Nl+1, 1:2*Nl:2] = np.diag((0, ) + (1, )*(Nl-1))
        else :  
            coefs[ku-3*Nl+1+0:ku-Nl+1:2, beg_j+0:beg_j+2*Nl:2] = -cpl.boundary[D[0]]
            coefs[ku-3*Nl+1+1:ku-Nl+1:2, beg_j+1:beg_j+2*Nl:2] = -np.eye(Nl)
        
        # Outer boundary conditions
        if d == domains.n_domains - 1 : 
            coefs[ku-Nl+1:ku+1, -2*Nl+0::2] = np.eye(Nl)
            coefs[ku-Nl+1:ku+1, -2*Nl+1::2] = np.diag((l+1)/2)
        else :  
            coefs[ku-Nl+1+0:ku+Nl+1:2, end_j-2*Nl+0:end_j:2] = cpl.boundary[D[-1]]
            coefs[ku-Nl+1+1:ku+Nl+1:2, end_j-2*Nl+1:end_j:2] = np.eye(Nl)
           
    col_scale = np.ones((Mb)) 
    if rescale :
        # Finding the row scaling factors
        row_max = np.abs(coefs.reshape((2*KLAG*2*Nl, N+NE, 2*Nl))).max(axis=2)
        offset = (2*KLAG-1)*Nl
        row_scale = np.zeros((Mb+2*offset))
        for j in range(N+NE) :
            j0 = 2*Nl*j
            row_scale[j0:j0+dj] = np.maximum(row_scale[j0:j0+dj], row_max[:, j])
        row_scale = np.divide(
            1.0, row_scale, out=np.zeros((Mb+2*offset)), where=row_scale!=0.0
        )
        
        # Applying the row scales
        b *= row_scale[offset:-offset]
        R_scale2D = np.array([row_scale[2*Nl*j:2*Nl*j+dj] for j in range(N+NE)]).T
        coefs = (
            coefs.reshape((-1, N+NE, 2*Nl)) * R_scale2D[..., None]
        ).reshape(coefs.shape)
        
        # Finding the column scaling factors
        col_scale = 1.0 / np.abs(coefs).max(axis=0)
        
        # Applying the column scales
        coefs *= col_scale
            
    return coefs, b, col_scale


def find_phi_eff(r2d, t, rho, phi_eff=None, rescale_ab=True) :
    """
    Determination of the effective potential from a given mapping
    (mapping, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    r2d : array_like, shape (N, M)
        Current mapping.
    t : array_like, shape (M, )
        Values of cos(theta).
    rho : array_like, shape (N, )
        Current density on each equipotential.
    phi_eff : array_like, shape (N, ), optional
        If given, the current effective potential on each 
        equipotential. If not given, it will be calculated inside
        this fonction. The default is None.
    rescale_ab : boolean
        Whether to equilibrate Poisson's matrix with a rescaling before
        the LU decomposition.

    Raises
    ------
    ValueError
        If the matrix inversion enconters a difficulty ...

    Returns
    -------
    phi_g_l : array_like, shape (N, L), optional
        Gravitation potential harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
    dphi_eff : array_like, shape (N, ), optional
        Effective potential derivative with respect to zeta.

    """        
    # Internal mapping derivatives
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=KSPL,
        domain_ranges=domains.domain_ranges[:-1],
    )

    # Matter source: internal domain only
    r2rz_l = pl_project_2D(
        r2d**2 * der.r_z,
        L,
    ) / (np.arange(L) + 1 / 2)

    rhs_l = 4 * np.pi * rho[:, None] * r2rz_l[:, ::2]

    # Vacuum extension: used for the Poisson operator
    z_ext = zeta[domains.external_mask]
    r2d_ext, der_ext = extend_mapping(r2d, der, z_ext)

    # Coupling terms for the Poisson operator
    cpl = compute_poisson_couplings(
        r2d_ext,
        der_ext,
        t,
        max_degree=L,
        alpha=2,
    )
    
    # Vector and band matrix storage characteristics
    Nl = (L+1)//2
    kl = (2*KLAG + 1) * Nl - 1
    ku = (2*KLAG + 1) * Nl - 1
    
    # Determination of matrix blocs (and b) from coupling harmonics (and rhs harmonics)
    # Rescale the coefficients in both coefs and b if rescale_ab is True.
    coefs, b, col_scale = find_Poisson_coefs(kl, ku, cpl, rhs_l, rescale=rescale_ab)  
    
    # Matrix filling (credits to N. Fargette for this part)
    ab = np.zeros((2*kl+ku+1, 2*Nl*(N+NE)))
    mask = np.zeros((2*Nl*(N+NE), kl+ku+1), dtype=bool)
    for l in range(2*Nl) : mask[l::2*Nl, L-l:kl+ku+1-l] = 1
    (ab[kl:, :]).T[mask] = (coefs.T).flatten()
    del coefs, mask; collect()   
    
    # System solving (LAPACK)
    *_, x, info = dgbsv(kl, ku, ab, b)
    del ab, b; collect()
    x *= col_scale
    if info != 0 : 
        raise ValueError(
            "Problem with finding the gravitational potential. \n",
            "Info = ", info
        )
            
    # Poisson's equation solution
    phi_g_l  = np.zeros((N+NE, L))
    dphi_g_l = np.zeros((N+NE, L))
    phi_g_l[: , ::2] = x[1::2].reshape((N+NE, Nl))
    dphi_g_l[:, ::2] = x[0::2].reshape((N+NE, Nl))        
    
    if phi_eff is None :

        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        dphi_eff = pl_eval_2D(dphi_g_l, 0.0)        
        return phi_g_l, dphi_g_l, phi_eff, dphi_eff
        
    # The effective potential is known to an additive constant 
    C = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += C
    return phi_g_l, dphi_g_l, phi_eff


def find_new_mapping(r2d, t, omega_n, phi_g_l, dphi_g_l, phi_eff, dphi_eff) :
    """
    Find the new mapping by comparing the effective potential
    and the total potential (calculated from phi_g_l and omega_n).

    Parameters
    ----------
    mapping : array_like, shape (N, M)
        Current mapping.
    t : array_like, shape (M, )
        Value of cos(theta).
    omega_n : float
        Current rotation rate.
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (N, L)
        Gravitation potential derivative harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each level surface.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative over the isopotentials.

    Returns
    -------
    map_n_new : array_like, shape (N, M)
        Updated mapping.
    omega_n_new : float
        Updated rotation rate.

    """
    
    # Define the targets
    targets = np.copy(phi_eff[:N])
    
    # Find metric terms
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=KSPL,
        domain_ranges=domains.domain_ranges[:-1],
    )
    z_ext = zeta[domains.external_mask]
    r2d_ext, der_ext = extend_mapping(r2d, der, z_ext)
    
    # 2D gravitational potential
    eq = (M-1)//2
    up = np.arange(eq+1)
    phi2D_g = pl_eval_2D( phi_g_l, t[up])
    dphi2D_g_dz = pl_eval_2D(dphi_g_l, t[up])
    dphi2D_g = dphi2D_g_dz / der_ext.r_z[:, up]
        
    ### Find the adaptive rotation rate
    valid_z = zeta > 0.5
    valid_r = r2d_ext[valid_z, eq]
    phi1D_c, dphi1D_c = eval_phi_c(valid_r, 0.0, omega_n) / valid_r ** 3
    dphi1D_c -= 3 * phi1D_c / valid_r
    phi1D  =  phi2D_g[valid_z, eq] +  phi1D_c
    dphi1D = dphi2D_g[valid_z, eq] + dphi1D_c
    
    unq_r = find_domains(valid_r).unique_indices
    r_est = CubicHermiteSpline(
        x=phi1D[unq_r], y=valid_r[unq_r], dydx=dphi1D[unq_r] ** -1
    )(targets[-1])
    omega_n_new = omega_n * r_est**(-1.5)
                
    ### Find the new mapping using the reciprocal interpolation
    # Define the adaptive mesh
    new_res = int(1.0/(np.finfo(float).eps)**0.2)
    d2phi_eff = np.hstack((0.0, np.abs(np.diff(dphi_eff[domains.unique_indices]))))
    z_new = interpolate_func(d2phi_eff.cumsum(), zeta[domains.unique_indices], k=1)(
        np.linspace(0.0, d2phi_eff.sum(), new_res)
    )
    z_new = 2 * (z_new - z_new[0]) / (z_new[-1] - z_new[0])
    
    # Cubic Hermite splines
    r_splines = [CubicHermiteSpline(
        x=zeta[domains.unique_indices], y=r2d_ext[domains.unique_indices, k], dydx=der_ext.r_z[domains.unique_indices, k]
    ) for k in up]
    p_splines = [CubicHermiteSpline(
        x=zeta[domains.unique_indices], y=phi2D_g[domains.unique_indices, k], dydx=dphi2D_g_dz[domains.unique_indices, k]
    ) for k in up]
    
    # Interpolated variables
    r_ipl, dr_ipl = [
        np.array([r_spl(z_new, nu=nu) for r_spl in r_splines]).T for nu in [0, 1]
    ]
    phi_g_ipl, dphi_g_ipl = [
        np.array([p_spl(z_new, nu=nu) for p_spl in p_splines]).T for nu in [0, 1]
    ]
    phi_c_ipl, dphi_c_ipl = np.moveaxis(np.array([
        eval_phi_c(rk , ck, omega_n_new) for rk, ck in zip(r_ipl[:, up].T, t[up])
    ]), 0, 2)
    phi_ipl  =  phi_c_ipl +  phi_g_ipl
    dphi_ipl = dphi_c_ipl + dphi_g_ipl / dr_ipl
    
    # Finding the valid interpolation domain
    valid = valid_reciprocal_domain(z_new, dphi_ipl)
        
    # Estimate at target values
    map_est = np.zeros_like(r2d[:, up])
    map_est[1:] = np.array([
        CubicHermiteSpline(x=pk[vk], y=rk[vk], dydx=dpk[vk]**-1)(targets[1:]) 
        for rk, pk, dpk, vk in zip(r_ipl.T, phi_ipl.T, dphi_ipl.T, valid.T)
    ]).T
    map_est[domains.interface_start_indices[:-1]] = map_est[domains.interface_end_indices[:-1]]
        
    ### New mapping
    map_n_new = np.hstack((map_est, np.flip(map_est, axis=1)[:, 1:]))
        
    return map_n_new, omega_n_new


def Virial_theorem(r2d, rho, omega_n, phi_g_l, P, verbose=False) : 
    """
    Compute the Virial equation and gives the resukt as a diagnostic
    for how well the hydrostatic equilibrium is satisfied (the closer
    to zero, the better).
    
    Parameters
    ----------
    mapping : array_like, shape (N, M)
        Mapping
    rho : array_like, shape (N, )
        Density on each equipotential.
    omega_n : float
        Rotation rate.
    phi_g_l : array_like, shape (N, L)
        Gravitational potential harmonics.
    P : array_like, shape (N, )
        Pressure on each equipotential.
    verbose : bool
        Whether to print the individual energy values or not.
        The default is None.

    Returns
    -------
    virial : float
        Value of the normalised Virial equation.

    """    
    # Potential energy
    volumic_potential_energy = lambda rk, ck, D : -rho[D] * pl_eval_2D(phi_g_l[D], ck)
    potential_energy = integrate2D(
        r2d, volumic_potential_energy, domains=domains.domain_ranges[:-1], k=KSPL
    )
    
    # Kinetic energy
    volumic_kinetic_energy = lambda rk, ck, D : (  
       0.5 * rho[D] * (1-ck**2) * rk[D]**2 * eval_omega(rk[D], ck, omega_n)**2
    )
    kinetic_energy = integrate2D(
        r2d, volumic_kinetic_energy, domains=domains.domain_ranges[:-1], k=KSPL
    )
    
    # Internal energy
    internal_energy = integrate2D(r2d, P, domains=domains.domain_ranges[:-1], k=KSPL)
    
    # Surface term
    _, weights = roots_legendre(M)
    surface_term = 2*np.pi * (r2d[-1]**3 @ weights) * P[-1]
    
    # Compute the virial equation
    if verbose :
        print(f"Kinetic energy  : {kinetic_energy:12.10f}")
        print(f"Internal energy : {internal_energy:12.10f}")
        print(f"Potential energy: {potential_energy:12.10f}")
        print(f"Surface term    : {surface_term:12.10f}")
    virial = ( 
          (2*kinetic_energy - 0.5*potential_energy + 3*internal_energy - surface_term)
        / (2*kinetic_energy + 0.5*potential_energy + 3*internal_energy + surface_term)
    )
    print(f"Virial theorem verified at {round(virial, 16)}")
    return virial


def solve_spheroidal(
    model: Model1D,
    rotation_config: RotationConfig,
    solver_options: SolverOptions,
    output_options: OutputOptions,
) -> SpheroidalResult:
    """
    Compute rotational deformation on a multidomain spheroidal grid.

    Poisson's equation is solved directly in the evolving material
    coordinates, extended by an exterior vacuum domain. The mapping,
    rotation state, mass, and radius are iterated until convergence.
    """
    global N, NE, L, M, KSPL, KLAG
    global zeta, domains
    global Lsp, Dsp
    global eval_phi_c, eval_omega

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
    I = num.n_internal_points
    I_ext = num.n_external_points
    J = num.angular_resolution

    # Temporary aliases required by legacy helpers
    N = I
    NE = I_ext
    M = J

    L = num.max_degree
    KSPL = num.spline_order
    KLAG = num.lagrange_order

    zeta = num.zeta
    domains = num.domains
    Lsp = num.Lsp
    Dsp = num.Dsp

    # Temporary evaluators required by update_mapping and the virial
    eval_phi_c, eval_omega = configure_rotation_profile(
        rotation_config.profile,
        rotation_config.central_diff_rate,
        rotation_config.scale,
    )

    rescale_ab = solver_options.rescale_ab
    full_rate = solver_options.full_rate
    mapping_precision = solver_options.mapping_precision
    max_iterations = solver_options.max_iterations
    rotation_target = rotation_config.target
    
    # Angular domain preparation
    r2d, t = initialize_mapping(r1d, M)
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff = find_phi_eff(r2d, t, rho, rescale_ab=rescale_ab)
    
    # Find pressure
    p = find_pressure(
        rho,
        dphi_eff,
        surface_pressure,
    )
    
    # Iterative centrifugal deformation
    polar_radius_history = [0.0, find_r_pol(r2d, L)]
    iterations = 0
    print(
        "\n+---------------------+",
        "\n| Deformation started |", 
        "\n+---------------------+\n"
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
                "Spheroidal deformation did not converge after "
                f"{max_iterations} iterations. "
                f"Last |delta R_pol| = "
                f"{delta_polar:.3e}, "
                f"target = {mapping_precision:.3e}. "
                f"Recent polar radii: "
                f"{recent_radii!r}"
            )
        
        # Current rotation rate
        omega_n = min(rotation_target, ((iterations+1)/full_rate) * rotation_target)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(r2d, t, rho, phi_eff, rescale_ab)
        
        # Update the mapping
        r2d, omega_n = find_new_mapping(
            r2d, t, omega_n, phi_g_l, dphi_g_l, phi_eff, dphi_eff
        )        

        # Renormalisation
        r_corr    = find_r_eq(r2d, L)
        m_corr    = integrate2D(r2d, rho, domains=domains.domain_ranges[:-1])   
        radius   *= r_corr
        mass     *= m_corr
        r2d      /=             r_corr
        rho      /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr    # <- /!\ This is a derivative w.r.t. to zeta
        p        /= m_corr**2 / r_corr**4
        
        # Update the polar radius
        polar_radius_history.append(find_r_pol(r2d, L))
        
        # Iteration count
        iterations += 1
        DEC = int(-np.log10(mapping_precision))
        print(f"Iteration n°{iterations:02d}, R_pol = {polar_radius_history[-1].round(DEC)}")
        
    finish = time.perf_counter()
    print(
        "\n+------------------+",
        "\n| Deformation done |", 
        "\n+------------------+\n"
    )
    print(f'Time taken: {round(finish-start, 2)} secs')  
    
    # Estimated error on Poisson's equation
    der = compute_mapping_derivatives(
        r2d,
        zeta,
        t,
        max_degree=L,
        spline_order=KSPL,
        domain_ranges=domains.domain_ranges[:-1],
    )
    z_ext = zeta[domains.external_mask]
    r2d_ext, der_ext = extend_mapping(r2d, der, z_ext)
    
    # Store the normalised solver state before any output
    # operation can modify the arrays in place.
    result = SpheroidalResult(
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
        rotation_rate=omega_n,

        polar_radius_history=np.asarray(polar_radius_history),
        iterations=iterations,

        internal_zeta=zeta[domains.internal_mask].copy(),
        external_zeta=zeta[domains.external_mask].copy(),
        full_mapping=r2d_ext.copy(),

        internal_mask=domains.internal_mask.copy(),
        external_mask=domains.external_mask.copy(),
    )
    
    
    if output_options.plot.show_harmonics :
        phi_g_harmonics(zeta, phi_g_l, radial=False)
    
    # Virial test
    if output_options.diagnostics.virial_test :
        virial = Virial_theorem(r2d, rho, omega_n, phi_g_l, p, verbose=True)   
    
    # Plot model
    if output_options.plot.show_model :
        plot_f_map(
            r2d, np.log10(rho+rho.max()**-1), phi_eff, L, 
            angular_res=output_options.plot.resolution,
            cmap=output_options.plot.field_cmap,
            show_surfaces=output_options.plot.surfaces,
            cmap_lines=output_options.plot.surface_cmap,
            disc=domains.interface_end_indices[:-1],
            label=r"$\log_{10} \left[\rho \times {\left(M/R_{\mathrm{eq}}^3\right)}^{-1}\right]$"
        )
    
    # Gravitational moments
    if output_options.diagnostics.gravitational_moments :
        find_gravitational_moments(r2d, rho)
    
    # Model writing
    if output_options.model.save :
        rota = eval_omega(r2d[:, (M-1)//2], 0.0, rotation_target)
        if output_options.model.dimensional : 
            r2d  *=               radius
            rho      *=     mass    / radius**3
            phi_eff  *= G * mass    / radius   
            dphi_eff *= G * mass    / radius
            p        *= G * mass**2 / radius**4
        write_model(
            output_options.model.filename,
            (N, M, mass, radius, rotation_target, G),
            r2d,
            additional_variables,
            zeta,
            p,
            rho,
            phi_eff,
            rota,
        )
    
    return result
        
    
    
    
    

    