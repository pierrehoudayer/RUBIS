import time
import numpy        as np
import scipy.sparse as sps
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.special       import roots_legendre, eval_legendre
from scipy.integrate     import solve_ivp

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
from ..models            import PolytropicModelConfig
from ..polytrope         import build_polytrope
from ..mapping           import (
    initialize_mapping, 
    valid_reciprocal_domain,
    compute_mapping_derivatives,
    compute_mapping_geometry,
)
from ..rotation_profiles import configure_rotation_profile
from ..results           import RadialResult
from ..io.legacy         import write_model
from ..plotting          import (
    phi_g_harmonics,
    plot_3D_surface,
    plot_f_map,
    plot_flux_lines,
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
    mass : float
        Total mass of the model.
    radius : float
        Radius of the model.
    r1d : array_like, shape (N, ), [GLOBAL VARIABLE]
        Radial coordinate after normalisation.
    zeta : array_like, shape (N, ), [GLOBAL VARIABLE]
        Spheroidal coordinate
    rho : array_like, shape (N, )
        Radial density of the model after normalisation.
    additional_variables : array_like, shape (N, N_var)
        Additional variables found in 'MOD1D'.

    """
    G = 6.67384e-8  # <- Gravitational constant
    if isinstance(model_choice, PolytropicModelConfig): 
        # The model properties are user-defined
        N      = model_choice.n_points
        mass   = model_choice.mass
        radius = model_choice.radius
        
        # Polytrope computation
        model = build_polytrope(model_choice)        
        
        # Normalisation
        r1d = model.r     / (              radius   )
        rho = model.rho   / (    mass    / radius**3)
        P0  = model.p[-1] / (G * mass**2 / radius**4)
        additional_variables = []
        
    else : 
        # Reading file 
        surface_pressure, radial_res = np.genfromtxt(
            './Models/'+model_choice, max_rows=2, unpack=True
        )
        r1d, rho1d, *additional_variables = np.genfromtxt(
            './Models/'+model_choice, skip_header=2, unpack=True
        )
        _, idx = np.unique(r1d, return_index=True) 
        N = len(idx)
        
        # Normalisation
        radius = r1d[-1]
        mass = 4*np.pi * integrate(x=r1d[idx], y=r1d[idx]**2 * rho1d[idx])
        r1d =   r1d[idx]       / (          radius   )
        rho = rho1d[idx]       / (mass    / radius**3)
        P0  = surface_pressure / (mass**2 / radius**4)
        
        # We assume that P0 is already normalised by G if R ~ 1 ...
        if not np.allclose(radius, 1) :
            P0 /= G
            
    zeta = np.copy(r1d)
    
    return G, P0, N, mass, radius, r1d, zeta, rho, additional_variables


def init_sparse_matrices() : 
    """
    Finds the sparse matrices used for filling Poisson's matrix.

    Returns
    -------
    Lsp, Dsp, Asp : sparse matrices in DIAgonal format (see scipy.sparse)
        Sparse matrices respectively storing the interpolation (Lsp) and
        derivation (Dsp) coefficients and the derivative -> derivative
        terms of Poisson's matrix.

    """
    lag_mat = lagrange_matrix_P(r1d**2, order=KLAG)
    Lsp = sps.dia_matrix(lag_mat[..., 0])
    Dsp = sps.dia_matrix(lag_mat[..., 1])
    Asp = sps.dia_matrix(
        4 * lag_mat[..., 1] * r1d**4 - 2 * lag_mat[..., 0] * r1d**2
    )
    return Lsp, Dsp, Asp

def find_pressure(rho, dphi_eff, P0) :
    """
    Find the pressure evaluated on r1d thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : array_like, shape (N, )
        Density profile.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative with respect to r1d.
    P0 : float
        Surface pressure.

    Returns
    -------
    P : array_like, shape (N, )
        Pressure profile.

    """
    dP = - rho * dphi_eff
    P  = interpolate_func(1-r1d[::-1], -dP[::-1], der=-1, k=KSPL, prim_cond=(0, P0))(1-r1d[::-1])[::-1]
    return P


def find_rho_l(r2d, rho) : 
    """
    Find the density distribution harmonics from a given mapping 
    (r2d) which gives the lines of constant density (= rho).

    Parameters
    ----------
    r2d : array_like, shape (N, M)
        Current mapping.
    rho : array_like, shape (N, )
        Current density on each equipotential.

    Returns
    -------
    rho_l : array_like, shape (N, L)
        Density distribution harmonics.

    """
    # Density interpolation on the mapping
    safety_constant = 1e-15
    all_k   = np.arange((M+1)//2)
    log_rho = np.log(rho + safety_constant)
    rho2D   = np.zeros((N, M))
    for k in all_k :
        inside =  r1d < r2d[-1, k]
        rho2D[inside, k] = interpolate_func(
            x=r2d[:, k], y=log_rho, k=KSPL
        )(r1d[inside])
        rho2D[inside, k] = np.exp(rho2D[inside, k]) - safety_constant
    rho2D[:,-1-all_k] = rho2D[:, all_k]
    
    # Corresponding harmonic decomposition
    rho_l = pl_project_2D(rho2D, L)
    
    return rho_l

def filling_ab(ab, ku, kl, l) : 
    """
    Fill the band storage matrix according to the Poisson's
    equation (written in terms of r^2). We optimize this
    operation using the scipy sparse matrix storage (the 
    [::-1] comes from the opposite storage convention 
    between Fortran and Scipy) and exploiting the fact that 
    most of the matrix stay the same when changing l.

    Parameters
    ----------
    ab : array_like, shape (ldmat, 2*N)
        Band storage matrix.
    ku : integer
        Number of terms in the upper matrix part.
    kl : integer
        Number of terms in the lower matrix part.
    l : integer
        Harmonic degree.

    Returns
    -------
    ab : array_like, shape (ldmat, 2*N)
        Filled band storage matrix.

    """    
    # Offset definition
    offset = ku + kl
    
    # The common filling part
    if l == 0 :
        ab[ku+1+(0-0):-1+(0-0):2, 0::2] =  Asp.data[::-1]
        ab[ku+1+(1-0):        :2, 0::2] = -Lsp.data[::-1]
        ab[ku+1+(1-1):-1+(1-1):2, 1::2] =  Dsp.data[::-1]
                
        # First boundary condition (l = 0)
        ab[offset, 0] = 6.0
        
    # The only l dependent part (~1/4 of the matrix)
    else : 
        ab[ku+1+(0-1):-1+(0-1):2, 1::2] = -l*(l+1) * Lsp.data[::-1]
        
        # First boundary condition (l != 0)
        ab[offset-0, 0] = 0.0
        ab[offset-1, 1] = 1.0
    
    # Boundary conditions
    ab[offset+1, 2*N-2] = 2*r1d[-1]**2
    ab[offset+0, 2*N-1] = l+1 
    return ab
    
    
def find_phi_eff(r2d, rho, phi_eff=None, lub_l=None) :
    """
    Determination of the effective potential from a given mapping
    (r2d, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    r2d : array_like, shape (N, M)
        Current mapping.
    rho : array_like, shape (N, )
        Density on each equipotential.
    phi_eff : array_like, shape (N, ), optional
        If given, the current effective potential on each 
        equipotential. If not given, it will be calculated inside
        this fonction. The default is None.
    lub_l : list (size: Nl) of tuples (size: 2), optional
        Each element of the list contains contains the LU
        decomposition of Poisson's matrix (first tuple element)
        and the corresponding pivot indices (second tuple element)
        to solve Poisson's equation at a diven degree l. The 
        routine therefore fully exploit the invariance of Poisson's 
        matrix by computing those two elements once and for all.
        The default is None.

    Raises
    ------
    ValueError
        If the matrix inversion enconters a difficulty ...

    Returns
    -------
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics derivative with respect to r^2.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
    dphi_eff : array_like, shape (N, ), optional
        Effective potential derivative with respect to r^2.
    lub_l : list (size: Nl) of tuples (size: 2), optional
        Cf. parameters

    """    
    # Density distribution harmonics
    rho_l    = find_rho_l(r2d, rho)
    phi_g_l  = np.zeros((N, L))
    dphi_g_l = np.zeros((N, L))
    
    # Vector filling (vectorial)
    Nl = (L+1)//2
    bl = np.zeros((2*N, Nl))
    bl[1:-1:2, :] = 4*np.pi * Lsp @ (r1d[:,None]**2 * rho_l[:, ::2])
    bl[0     , 0] = 4*np.pi * rho_l[0, 0]     # Boundary condition
    
    # Band matrix storage
    kl = 2*KLAG
    ku = 2*KLAG
    ab = np.zeros((2*kl + ku + 1, 2*N))   
    
    if phi_eff is None :
        lub_l = []
        for l in range(0, L, 2) :
            # Matrix filling  
            ab = filling_ab(ab, ku, kl, l)
            
            # LU decomposition (LAPACK)
            lub_l.append(dgbtrf(ab, ku, kl)[:-1])
            
    # System solving (LAPACK)
    x = np.array([
        dgbtrs(lub_l[k][0], kl, ku, bl[:, k], lub_l[k][1])[0] for k in range(Nl)
    ]).T
        
    # Poisson's equation solution
    phi_g_l[: , ::2] = x[1::2]
    dphi_g_l[:, ::2] = x[0::2] * (2*r1d[:, None])  # <- The equation is solved on r^2
    
    if phi_eff is None :
        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        dphi_eff = pl_eval_2D(dphi_g_l, 0.0)        
        return phi_g_l, dphi_g_l, phi_eff, dphi_eff, lub_l
    
    # The effective potential is known to an additive constant 
    C = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += C
    return phi_g_l, dphi_g_l, phi_eff


def find_new_mapping(t, omega_n, phi_g_l, dphi_g_l, phi_eff) :
    """
    Find the new mapping by comparing the effective potential
    and the total potential (calculated from phi_g_l and omega_n).

    Parameters
    ----------
    t : array_like, shape (M, )
        Value of cos(theta).
    omega_n : float
        Current rotation rate.
    phi_g_l : array_like, shape (N, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (N, L)
        Gravitation potential derivative harmonics.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.

    Returns
    -------
    map_n_new : array_like, shape (N, M)
        Updated mapping.
    omega_n_new : float
        Updated rotation rate.

    """    
    # 2D gravitational potential (interior)
    eq = (M-1)//2
    up = np.arange(eq+1)
    phi2D_g_int  = pl_eval_2D( phi_g_l, t[up])
    dphi2D_g_int = pl_eval_2D(dphi_g_l, t[up])
    
    # 2D gravitational potential (exterior)
    l = np.arange(L)
    outside = 1.3        # Some guess
    r_ext = np.linspace(1.0, outside, 101)[1:]
    phi_g_l_ext  = phi_g_l[-1] * (r_ext[:, None])**-(l+1)
    dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_ext[:, None]
    phi2D_g_ext  = pl_eval_2D( phi_g_l_ext, t[up])
    dphi2D_g_ext = pl_eval_2D(dphi_g_l_ext, t[up])
    
    # 2D gravitational potential
    r_tot = np.hstack((r1d, r_ext))
    phi2D_g  = np.vstack(( phi2D_g_int,  phi2D_g_ext))
    dphi2D_g = np.vstack((dphi2D_g_int, dphi2D_g_ext))
        
    # Find a new value for ROT
    valid_z = r_tot > 0.5
    valid_r = r_tot[valid_z]
    phi1D_c, dphi1D_c = eval_phi_c(valid_r, 0.0, omega_n) / valid_r ** 3
    dphi1D_c -= 3 * phi1D_c / valid_r
    phi1D  =  phi2D_g[valid_z, eq] +  phi1D_c
    dphi1D = dphi2D_g[valid_z, eq] + dphi1D_c
    r_est = CubicHermiteSpline(x=phi1D, y=valid_r, dydx=dphi1D ** -1)(phi_eff[-1])
    omega_n_new = omega_n * r_est**(-1.5)
    
    # Centrifugal potential
    phi2D_c, dphi2D_c = np.moveaxis(np.array([
        eval_phi_c(r_tot , ck, omega_n_new) for ck in t[up]
    ]), (0, 1, 2), (2, 0, 1))
    
    # Total potential
    phi2D  =  phi2D_g +  phi2D_c
    dphi2D = dphi2D_g + dphi2D_c
    
    # Finding the valid interpolation domain
    valid = valid_reciprocal_domain(r_tot, dphi2D)
    
    # Central domain
    lim = 1e-1
    center = np.max(np.argwhere(r_tot < lim)) + 1
    r_cnt = np.linspace(0.0, 1.0, 5*center) ** 2 * lim
    phi2D_g_cnt = np.array([CubicHermiteSpline(
        x=r1d, y=phi2D_g_int[:, k], dydx=dphi2D_g_int[:, k]
    )(r_cnt) for k in up]).T
    phi2D_c_cnt = np.array([
        eval_phi_c(r_cnt , ck, omega_n_new)[0] for ck in t[up]
    ]).T
    phi2D_cnt = phi2D_g_cnt + phi2D_c_cnt
    
    # Estimate at target values
    map_est = np.vstack((
        np.zeros_like(up), 
        np.array([
            interpolate_func(x=pk, y=r_cnt, k=KSPL)(phi_eff[1:center]) 
            for pk in phi2D_cnt.T
        ]).T,
        np.array([
            interpolate_func(x=pk[vk], y=r_tot[vk], k=KSPL)(phi_eff[center:]) 
            for pk, vk in zip(phi2D.T, valid.T)
        ]).T
    ))
            
    # New mapping
    map_n_new = np.hstack((map_est, np.flip(map_est, axis=1)[:, 1:]))
        
    return map_n_new, omega_n_new


def Virial_theorem(r2d, rho, omega_n, phi_eff, P, verbose=False) : 
    """
    Compute the Virial equation and gives the result as a diagnostic
    for how well the hydrostatic equilibrium is satisfied (the closer
    to zero, the better).
    
    Parameters
    ----------
    r2d : array_like, shape (N, M)
        Mapping
    rho : array_like, shape (N, )
        Density on each equipotential.
    omega_n : float
        Rotation rate.
    phi_eff : array_like, shape (N, )
        Effective potential on each equipotential.
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
    volumic_potential_energy = lambda rk, ck, D : -(  
       rho[D] * (phi_eff[D]-eval_phi_c(rk[D], ck, omega_n)[0])
    )
    potential_energy = integrate2D(r2d, volumic_potential_energy, k=KSPL)
    
    # Kinetic energy
    volumic_kinetic_energy = lambda rk, ck, D : (  
       0.5 * rho[D] * (1 - ck**2) * rk[D]**2 * eval_omega(rk[D], ck, omega_n)**2
    )
    kinetic_energy = integrate2D(r2d, volumic_kinetic_energy, k=KSPL)
    
    # Internal energy
    internal_energy = integrate2D(r2d, P, k=KSPL)
    
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


def find_gravitational_moments(r2d, t, rho, max_degree=14) :
    """
    Find the gravitational moments up to max_degree.

    Parameters
    ----------
    r2d : array_like, shape (N, M)
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
            r2d, rho[:, None] * r2d ** l * eval_legendre(l, t), k=KSPL
        )
        print("Moment n°{:2d} : {:+.10e}".format(l, m_l))
        

def radial_method(*params, max_iterations=200) -> RadialResult:
    """
    Main routine for the centrifugal deformation method in radial coordinates.

    Parameters
    ----------
    params : tuple
        All method parameters. Please refer to the documentation in RUBIS.py

    """
    
    # Global parameters, constants, variables and functions
    start = time.perf_counter()
    global L, M, KSPL, KLAG, N, r1d, zeta, eval_phi_c, eval_omega, Lsp, Dsp, Asp
    model_choice, rotation_profile, rotation_target, central_diff_rate, \
    rotation_scale, L, M, full_rate, mapping_precision, KSPL, KLAG, output_options \
    , _, _ = params
    
    # Sanity check    
    if (
        isinstance(model_choice, PolytropicModelConfig)
        and model_choice.n_regions > 1
    ):
        raise ValueError(
            "The radial solver only supports single-domain models; "
            "use the spheroidal solver for composite polytropes."
        )
        
    # Definition of the 1D-model
    G, P0, N, mass, radius, r1d, zeta, rho, additional_variables = init_1D(model_choice)  
    
    
    # Angular domain initialisation
    r2d, t = initialize_mapping(r1d, M)
    
    # Centrifugal potential and profile definition
    eval_phi_c, eval_omega = configure_rotation_profile(
        rotation_profile, 
        central_diff_rate, 
        rotation_scale
    )
    
    # Define sparse matrices 
    Lsp, Dsp, Asp = init_sparse_matrices()
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff, lub_l = find_phi_eff(r2d, rho)
    
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
        omega_n = min(rotation_target, ((iterations+1)/full_rate) * rotation_target)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(r2d, rho, phi_eff, lub_l)

        # Find a new estimate for the mapping
        r2d, omega_n = find_new_mapping(t, omega_n, phi_g_l, dphi_g_l, phi_eff)
        
        # Renormalisation
        r_corr    = find_r_eq(r2d, L)
        m_corr    = integrate2D(r2d, rho, k=KSPL)
        radius   *= r_corr
        mass     *= m_corr
        r2d    /=             r_corr
        rho    /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr**2
        P        /= m_corr**2 / r_corr**4
        
        # Update the polar radius
        polar_radius_history.append(find_r_pol(r2d, L))
        
        # Iteration count
        iterations += 1
        DEC = int(-np.log10(mapping_precision))
        print(f"Iteration n°{iterations:02d}, R_pol = {polar_radius_history[-1].round(DEC)}")
    
    # Deformation summary
    finish = time.perf_counter()
    print(
        "\n+------------------+",
        "\n| Deformation done |", 
        "\n+------------------+\n"
    )
    print(f'Time taken: {round(finish-start, 2)} secs')  
    
    
    # Store the normalised solver state before any output
    # operation can modify the arrays in place.
    result = RadialResult(
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
    )

    # Estimated error on Poisson's equation
    if output_options.plot.show_harmonics : 
        phi_g_harmonics(zeta, phi_g_l, radial=True)
    
    # Virial test
    if output_options.diagnostics.virial_test : 
        virial = Virial_theorem(r2d, rho, omega_n, phi_eff, P, verbose=True)   
    
    # Plot model
    if output_options.plot.show_model :
        
        # Variable to plot
        f = rho
        label = r"$\rho \times {\left(M/R_{\mathrm{eq}}^3\right)}^{-1}$"
        rota2D = np.array([eval_omega(rk, ck, rotation_target) for rk, ck in zip(r2d.T, t)]).T
        # if rota2D.max() - rota2D.min() > 1e-2 : 
        #     f = np.log10(rota2D)
        #     label = r"$\log_{10} \left(\Omega/\Omega_K\right)$"
            
        if output_options.flux.enabled : 
            z0 = output_options.flux.origin
            M1 = output_options.flux.n_lines
            Q_l, (fig, ax) = find_radiative_flux(
                r2d, t, z0, M1,
                add_flux_lines=output_options.flux.plot_lines, 
                show_T_eff=output_options.flux.show_effective_temperature,
                res=output_options.flux.resolution,
                flux_cmap=output_options.flux.cmap
            )
            plot_f_map(
                r2d, f, phi_eff, L, 
                angular_res=output_options.plot.resolution,
                cmap=output_options.plot.field_cmap,
                show_surfaces=output_options.plot.surfaces,
                cmap_lines=output_options.plot.surface_cmap,
                label=label,
                add_to_fig=(fig, ax) if output_options.flux.plot_lines else None
            )
        else : 
            plot_f_map(
                r2d, f, phi_eff, L, 
                angular_res=output_options.plot.resolution,
                cmap=output_options.plot.field_cmap,
                show_surfaces=output_options.plot.surfaces,
                cmap_lines=output_options.plot.surface_cmap,
                label=label
            )      
    
    # Gravitational moments
    if output_options.diagnostics.gravitational_moments :
        find_gravitational_moments(r2d, t, rho)
    
    # Model writing
    if output_options.model.save :
        rota = eval_omega(r2d[:, (M-1)//2], 0.0, rotation_target)
        if output_options.model.dimensional : 
            r2d    *=               radius
            rho      *=     mass    / radius**3
            phi_eff  *= G * mass    / radius   
            dphi_eff *= G * mass    / radius**2
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

    
#----------------------------------------------------------------#
#                   Radiative flux computation                   #
#----------------------------------------------------------------#
def find_radiative_flux(
    mapping, t, z0, M_lines, 
    add_flux_lines, show_T_eff, res, flux_cmap
) :
    """
    Determines the radiative flux lines and the surface flux, given 
    a model mapping (mapping) and a boundary on which to impose a 
    constant flux (characteristed by z0).

    Parameters
    ----------
    mapping : array_like, shape (N, M)
        Model mapping.
    t : array_like, shape (M, )
        Angular variable.
    z0 : float
        Zeta value for which the radiative flux is assumed to be constant.
        The value of the constant is determined by setting the rescaling
        the integrated flux on the surface by the star's luminosity.
    M_lines : integer
        Number of flux lines to be computed.
    add_flux_lines : boolean
        Whether to return a figure containing the flux lines so that
        they may be plotted on top of plot_f_map().
    show_T_eff : boolean
        Whether to show the effective temperature instead of the radiative
        flux amplitude on the 3D surface.
    res : tuple of floats (res_t, res_p)
        Gives the resolution of the 3D surface in theta and phi coordinates 
        respectively.
    flux_cmap : Colormap instance
        Colormap to plot the radiative flux at the model surface.

    Returns
    -------
    Q_l : array_like, shape (2*M_lines, )
        Radiative flux harmonics
    (fig, ax) : Subplot object
        Figure and axis containing the flux lines.

    """
    # Find domain
    from scipy.special import roots_legendre
    t1, w1 = np.array(roots_legendre(2*M_lines))[:, :M_lines]
    valid = np.squeeze(np.argwhere(zeta >= z0))
    z = zeta[valid]
    r = r2d[valid]
    x = (1 - z)[::-1]
    
    # Metric terms computation
    der = compute_mapping_derivatives(
        r,
        z,
        t,
        max_degree=L,
        spline_order=KSPL,
        domain_ranges=(slice(None),),
    )
    geo = compute_mapping_geometry(r, der, t)
    r_l   = pl_project_2D(r, L)
    rhs_l = pl_project_2D(geo.gg, L, even=False)
    jac_l = pl_project_2D(geo.jacobian, L)
    
    # Differential equation dt_dx = f(x, t)    
    def fun(xi, tk) : 
        rhs_t = np.atleast_2d(pl_eval_2D(rhs_l, tk.flatten()))
        f = np.array([
            interpolate_func(x, -rhs_tk[::-1], k=3)(xi) for rhs_tk in rhs_t.T
        ]).reshape(*tk.shape)
        return f
    
    def jac(xi, tk) : 
        jac_t = np.atleast_2d(pl_eval_2D(jac_l, tk.flatten()))
        J = np.diag([
            interpolate_func(x, -jac_tk[::-1], k=3)(xi) for jac_tk in jac_t.T
        ]).reshape(-1, *tk.shape)
        return J
    
    # Actual solving
    a = time.perf_counter()
    t = solve_ivp(
        fun=fun, 
        t_span=(0.0, 1-z0), 
        y0=t1, 
        method='LSODA', 
        dense_output=True, 
        rtol=1e-4, 
        atol=1e-4,
        jac=jac,
        vectorized=True
    ).sol(x).T[::-1]
    b = time.perf_counter()
    print(f"\nFlux lines found in {b-a:.2f} secs")
    
    # Integrate the flux along the characteristics
    r, r_t = np.moveaxis(
        np.array([pl_eval_2D(map_l[i], ti, der=1) for i, ti in enumerate(t)]), 0, 1
    )
    
    r_z_l      = pl_project_2D(der.r_z, L)
    divrel_z_l = pl_project_2D(geo.divrelz, L)
    r_z      = np.array([pl_eval_2D(     r_z_l[i], ti) for i, ti in enumerate(t)])
    divrel_z = np.array([pl_eval_2D(divrel_z_l[i], ti) for i, ti in enumerate(t)])
    
    Q_z = np.exp([-integrate(z, divrel_zk) for divrel_zk in divrel_z.T])
    Q_0 = 1.0 / sum(Q_z * (r[-1]**2 + (1-t1**2) * r_t[-1]**2) / r_z[-1] * w1)
    Q = Q_0 * Q_z * np.abs(
        pl_eval_2D(pl_project_2D(geo.gzz[-1], L), t[-1])
    )**0.5
    Q_l = pl_project_2D(np.hstack((Q, Q[::-1])), 2*M_lines)
    
    # 3D plot of the surface flux
    plot_3D_surface(r_l[-1], Q_l, show_T_eff=show_T_eff, res=res, cmap=flux_cmap)
    
    # Draw characteristics
    if add_flux_lines : 
        r = np.hstack((r, +r[:, ::-1]))
        t = np.hstack((t, -t[:, ::-1]))
        (fig, ax) = plot_flux_lines(r, t, color='grey')
    else : 
        (fig, ax) = (None, None)
    return Q_l, (fig, ax)