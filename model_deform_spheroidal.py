import time
import numpy             as np
import scipy.sparse      as sps
import scipy.special     as sp
from gc                  import collect
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbsv
from scipy.special       import roots_legendre, eval_legendre

from rubis.legendre      import (
    find_r_eq, 
    find_r_pol, 
    pl_eval_2D, 
    pl_project_2D,
    Legendre_coupling,
)
from rubis.numerical     import (
    integrate, 
    integrate2D, 
    interpolate_func, 
    lagrange_matrix_P,
)
from rubis.models        import PolytropicModelConfig
from rubis.polytrope     import build_polytrope
from rubis.domains       import find_domains
from rubis.mapping       import (
    initialize_mapping, 
    valid_reciprocal_domain,
    compute_mapping_derivatives,
    extend_mapping,
)
from rubis.poisson       import compute_poisson_couplings
from rubis.rotation_profiles import configure_rotation_profile
from rubis.results       import SpheroidalResult
from rubis.io.legacy     import write_model
from plot                import (
    plot_f_map, 
    phi_g_harmonics,
)

def init_1D(model_choice) : 
    """
    Function reading the 1D model file 'model_choice' (or generating a
    polytrope if model_choice is a dictionary). If additional variables are 
    found in the file, they are left unchanged and returned in the 
    output file.
    
    Parameters
    ----------
    model_choice : str or PolytropicModelConfig
        Filename of a spherical model or configuration of a generated
        polytropic model.

    Returns
    -------
    G : float
        Gravitational constant.
    P0 : float
        Value of the surface pressure after normalisation.
    N : integer
        Radial resolution of the model.
    M : float
        Total mass of the model.
    R : float
        Radius of the model.
    r1D : array_like, shape (N, ), [GLOBAL VARIABLE]
        Radial coordinate after normalisation.
    zeta : array_like, shape (N+NE, ), [GLrOBAL VARIABLE]
        Spheroidal coordinate
    rho : array_like, shape (N, )
        Radial density of the model after normalisation.
    other_var : array_like, shape (N, N_var)
        Additional variables found in 'MOD1D'.

    """
    G = 6.67384e-8  # <- Gravitational constant
    if isinstance(model_choice, PolytropicModelConfig):    
        # The model properties are user-defined
        N = model_choice.n_points
        M = model_choice.mass      
        R = model_choice.radius    
        
        # Polytrope computation
        model = build_polytrope(model_choice)
        
        # Normalisation
        r1d = model.r     /  R
        rho = model.rho   / (M/R**3)
        P0  = model.p[-1] / (G*M**2/R**4)
        other_var = np.empty_like(r1d)
        
    else : 
        # Reading file 
        surface_pressure, radial_res = np.genfromtxt(
            './Models/'+model_choice, max_rows=2, unpack=True
        )
        r1d, rho1d, *other_var = np.genfromtxt(
            './Models/'+model_choice, skip_header=2, unpack=True
        )
        N = int(radial_res)
        
        # Normalisationfor D in 
        R = r1d[-1]
        domains = find_domains(r1d)
        M = 4*np.pi * sum(
            integrate(x=r1d[D], y=r1d[D]**2 * rho1d[D]) for D in domains.domain_ranges
        )
        r1d = r1d / (R)
        rho = rho1d / (M/R**3)
        P0  = surface_pressure / (M**2/R**4)
        
        # We assume that P0 is already normalised by G if R ~ 1 ...
        if not np.allclose(R, 1) :
            P0 /= G
    
    # Spheroidal coordinate
    r1d_ext = np.linspace(1, 2, NE)
    zeta = np.hstack((r1d, 1 + sp.betainc(2, 2, r1d_ext-1)))
    
    return G, P0, N, M, R, r1d, zeta, rho, other_var


def init_sparse_matrices_per_domain() : 
    """
    Finds the interpolation and derivation matrices (in scipy sparse
    storage format) for each spheroidal domain.

    Returns
    -------
    Lsp, Dsp : list of array_like, shape (size_domain - 1, size_domain)
        Interpolation and derivation matrices.

    """
    Lsp, Dsp = [], []
    for D in domains.domain_ranges : 
        
        # Find the lagrange matrices per domain
        lag_mat = lagrange_matrix_P(zeta[D], order=KLAG)
        
        # Define sparse matrices 
        Lsp.append(sps.dia_matrix(lag_mat[..., 0]))
        Dsp.append(sps.dia_matrix(lag_mat[..., 1]))
    
    return Lsp, Dsp


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


def spheroidal_method(*params, max_iterations=200)-> SpheroidalResult:
    """
    Main routine for the centrifugal deformation method in spheroidal coordinates.

    Parameters
    ----------
    params : tuple
        All method parameters. Please refer to the documentation in RUBIS.py

    """
    
    # Global parameters, constants, variables and functions
    start = time.perf_counter()
    global L, M, KSPL, KLAG, NE, N, r1d, zeta, domains, eval_phi_c, eval_omega, Lsp, Dsp
    model_choice, rotation_profile, rotation_target, central_diff_rate, \
    rotation_scale, L, M, full_rate, mapping_precision, KSPL, KLAG, output_options, \
    NE, rescale_ab = params
    
    # Definition of the 1D-model
    G, P0, N, mass, radius, r1d, zeta, rho, additional_variables = init_1D(model_choice) 
    
    # Domains identification
    domains = find_domains(zeta)
    
    # Angular domain preparation
    r2d, t = initialize_mapping(r1d, M)
    
    # Centrifugal potential definition
    eval_phi_c, eval_omega = configure_rotation_profile(
        rotation_profile, 
        central_diff_rate, 
        rotation_scale
    )
    
    # Find the lagrange matrices per domain
    Lsp, Dsp = init_sparse_matrices_per_domain()
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff = find_phi_eff(r2d, t, rho, rescale_ab=rescale_ab)
    
    # Find pressure
    P = find_pressure(rho, dphi_eff, P0)
    
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
        r2d  /=             r_corr
        rho      /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr    # <- /!\ This is a derivative w.r.t. to zeta
        P        /= m_corr**2 / r_corr**4
        
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
        pressure=P.copy(),

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
        virial = Virial_theorem(r2d, rho, omega_n, phi_g_l, P, verbose=True)   
    
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
            P        *= G * mass**2 / radius**4
        write_model(
            output_options.model.filename,
            (N, M, mass, radius, rotation_target, G),
            r2d,
            additional_variables,
            zeta,
            P,
            rho,
            phi_eff,
            rota,
        )
    
    return result
        
    
    
    
    

    