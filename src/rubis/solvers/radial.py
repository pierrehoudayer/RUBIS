import time
import numpy        as np
import scipy.sparse as sps
from dataclasses         import dataclass
from numpy.typing        import NDArray
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbtrf, dgbtrs
from scipy.special       import roots_legendre, eval_legendre
from scipy.integrate     import solve_ivp

from ..config            import (
    OutputOptions,
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
from ..rotation_profiles import configure_rotation_profile
from ..results           import RadialResult
from ..io.legacy         import write_model
from ..plotting          import (
    phi_g_harmonics,
    plot_3D_surface,
    plot_f_map,
    plot_flux_lines,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True, kw_only=True)
class RadialNumerics:
    """Fixed numerical representation used by the radial solver."""

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
    """Build the fixed grids and operators of the radial solver."""
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

def find_pressure(
    rho,
    dphi_eff,
    surface_pressure,
    num: RadialNumerics,
):
    """Integrate hydrostatic equilibrium on the radial grid."""
    dp = -rho * dphi_eff
    x = 1.0 - num.r1d[::-1]

    p = interpolate_func(
        x=x,
        y=-dp[::-1],
        der=-1,
        k=num.spline_order,
        prim_cond=(0, surface_pressure),
    )(x)

    return p[::-1]


def find_rho_l(
    r2d,
    rho,
    num: RadialNumerics,
):
    """Project the density distribution onto Legendre harmonics."""
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
    
    
def filling_ab(ab, ku, kl, l, num: RadialNumerics):
    """Fill the band matrix of Poisson's equation."""
    I = num.n_points
    offset = ku + kl

    # Common part, filled only once.
    if l == 0:
        ab[ku+1+(0-0):-1+(0-0):2, 0::2] =  num.Asp.data[::-1]
        ab[ku+1+(1-0):        :2, 0::2] = -num.Lsp.data[::-1]
        ab[ku+1+(1-1):-1+(1-1):2, 1::2] =  num.Dsp.data[::-1]

        # Central boundary condition for l = 0.
        ab[offset, 0] = 6.0

    # Degree-dependent part.
    else:
        ab[ku+1+(0-1):-1+(0-1):2, 1::2] = -l*(l+1) * num.Lsp.data[::-1]

        # Central boundary condition for l != 0.
        ab[offset-0, 0] = 0.0
        ab[offset-1, 1] = 1.0

    # Surface boundary conditions.
    ab[offset+1, 2*I-2] = 2*num.r1d[-1]**2
    ab[offset+0, 2*I-1] = l+1 

    return ab

    
def find_phi_eff(
    r2d,
    rho,
    num: RadialNumerics,
    phi_eff=None,
    lub_l=None,
):
    """
    Determination of the effective potential from a given mapping
    (r2d, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    r2d : array_like, shape (I, J)
        Current mapping.
    rho : array_like, shape (I, )
        Density on each equipotential.
    phi_eff : array_like, shape (I, ), optional
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
    phi_g_l : array_like, shape (I, L)
        Gravitation potential harmonics.
    dphi_g_l : array_like, shape (I, L)
        Gravitation potential harmonics derivative with respect to r^2.
    phi_eff : array_like, shape (I, )
        Effective potential on each equipotential.
    dphi_eff : array_like, shape (I, ), optional
        Effective potential derivative with respect to r^2.
    lub_l : list (size: Nl) of tuples (size: 2), optional
        Cf. parameters

    """    
    I = num.n_points
    L = num.max_degree
    r1d = num.r1d[:, None]
    
    # Density distribution harmonics
    rho_l    = find_rho_l(r2d, rho, num)
    phi_g_l  = np.zeros((I, L))
    dphi_g_l = np.zeros((I, L))
    
    # Vector filling (vectorial)
    L_even = (L + 1) // 2
    b_l = np.zeros((2*I, L_even))
    b_l[1:-1:2, :] = 4 * np.pi * num.Lsp @ (r1d**2 * rho_l[:, ::2])
    b_l[0     , 0] = 4 * np.pi * rho_l[0, 0]     # Boundary condition
    
    # Band matrix storage
    kl = ku = 2 * num.lagrange_order
    ab = np.zeros((2*kl + ku + 1, 2*I))   
    
    if phi_eff is None :
        lub_l = []
        for l in range(0, L, 2) :
            # Matrix filling  
            ab = filling_ab(ab, ku, kl, l, num)
            
            # LU decomposition (LAPACK)
            lub_l.append(dgbtrf(ab, ku, kl)[:-1])
            
    # System solving (LAPACK)
    x = np.array([
        dgbtrs(lub_l[l][0], kl, ku, b_l[:, l], lub_l[l][1])[0] for l in range(L_even)
    ]).T
        
    # Poisson's equation solution
    phi_g_l[: , ::2] = x[1::2]
    dphi_g_l[:, ::2] = x[0::2] * (2*r1d)  # <- The equation is solved on r^2
    
    if phi_eff is None :
        # First estimate of the effective potential and its derivative
        phi_eff  = pl_eval_2D( phi_g_l, 0.0)
        dphi_eff = pl_eval_2D(dphi_g_l, 0.0)        
        return phi_g_l, dphi_g_l, phi_eff, dphi_eff, lub_l
    
    # The effective potential is known to an additive constant 
    C = pl_eval_2D(phi_g_l[0], 0.0) - phi_eff[0]
    phi_eff += C
    return phi_g_l, dphi_g_l, phi_eff


def find_new_mapping(
    omega,
    phi_g_l,
    dphi_g_l,
    phi_eff,
    num: RadialNumerics,
    eval_phi_c,
):
    """Update the mapping and rotation rate from the total potential."""
    r = num.r1d
    t = num.t
    L = num.max_degree
    J = num.angular_resolution
    k = num.spline_order

    # Northern hemisphere, including the equator.
    eq = (J - 1) // 2
    j_dw = np.arange(eq + 1)

    # Interior gravitational potential.
    phi2D_g_int  = pl_eval_2D( phi_g_l, t[j_dw])
    dphi2D_g_int = pl_eval_2D(dphi_g_l, t[j_dw])

    # Exterior gravitational potential.
    l = np.arange(L)
    outside = 1.3
    r_ext = np.linspace(1.0, outside, 101)[1:]

    phi_g_l_ext = phi_g_l[-1] * r_ext[:, None] ** -(l + 1)
    dphi_g_l_ext = -(l + 1) * phi_g_l_ext / r_ext[:, None]

    phi2D_g_ext  = pl_eval_2D( phi_g_l_ext, t[j_dw])
    dphi2D_g_ext = pl_eval_2D(dphi_g_l_ext, t[j_dw])

    # Full radial domain.
    r_tot = np.hstack((r, r_ext))
    phi2D_g  = np.vstack(( phi2D_g_int,  phi2D_g_ext))
    dphi2D_g = np.vstack((dphi2D_g_int, dphi2D_g_ext))

    # Find a rotation rate consistent with the equatorial radius.
    safe = r_tot > 0.0
    r_safe = r_tot[safe]

    phi1D_c, dphi1D_c = eval_phi_c(r_safe, 0.0, omega) / r_safe**3
    dphi1D_c -= 3.0 * phi1D_c / r_safe

    phi1D  =  phi2D_g[safe, eq] + phi1D_c
    dphi1D = dphi2D_g[safe, eq] + dphi1D_c

    r_est = CubicHermiteSpline(x=phi1D, y=r_safe, dydx=dphi1D**-1)(phi_eff[-1])
    omega_new = omega * r_est**-1.5

    # Centrifugal potential.
    phi2D_c, dphi2D_c = np.moveaxis(
        np.array([
            eval_phi_c(r_tot, ck, omega_new)
            for ck in t[j_dw]
        ]),
        (0, 1, 2),
        (2, 0, 1),
    )

    # Total potential.
    phi2D  =  phi2D_g + phi2D_c
    dphi2D = dphi2D_g + dphi2D_c

    valid = valid_reciprocal_domain(r_tot, dphi2D)

    # Refined central domain.
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
        eval_phi_c(r_cnt, ck, omega_new)[0]
        for ck in t[j_dw]
    ]).T

    phi2D_cnt = phi2D_g_cnt + phi2D_c_cnt

    # Estimate the radius at each target equipotential.
    r2d_up_origin = np.zeros_like(j_dw)
    r2d_up_center = np.array([
        interpolate_func(x=pk, y=r_cnt, k=k)(phi_eff[1:lim_idx])
        for pk in phi2D_cnt.T
    ]).T
    r2d_up_envelope = np.array([
        interpolate_func(x=pk[vk], y=r_tot[vk], k=k)(phi_eff[lim_idx:])
        for pk, vk in zip(phi2D.T, valid.T)
    ]).T
    r2d_up = np.vstack((
        r2d_up_origin,
        r2d_up_center,
        r2d_up_envelope,
    ))
    r2d_dw = np.flip(r2d_up, axis=1)[:, 1:]

    r2d_new = np.hstack((r2d_up, r2d_dw))

    return r2d_new, omega_new


def Virial_theorem(r2d, rho, omega_n, phi_eff, P, verbose=False) : 
    """
    Compute the Virial equation and gives the result as a diagnostic
    for how well the hydrostatic equilibrium is satisfied (the closer
    to zero, the better).
    
    Parameters
    ----------
    r2d : array_like, shape (I, J)
        Mapping
    rho : array_like, shape (I, )
        Density on each equipotential.
    omega_n : float
        Rotation rate.
    phi_eff : array_like, shape (I, )
        Effective potential on each equipotential.
    P : array_like, shape (I, )
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
    potential_energy = integrate2D(r2d, volumic_potential_energy, k=spl_order)
    
    # Kinetic energy
    volumic_kinetic_energy = lambda rk, ck, D : (  
       0.5 * rho[D] * (1 - ck**2) * rk[D]**2 * eval_omega(rk[D], ck, omega_n)**2
    )
    kinetic_energy = integrate2D(r2d, volumic_kinetic_energy, k=spl_order)
    
    # Internal energy
    internal_energy = integrate2D(r2d, P, k=spl_order)
    
    # Surface term
    _, weights = roots_legendre(J)
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
    r2d : array_like, shape (I, J)
        Isopotential mapping.
    t : array_like, shape (J, )
        Value of cos(theta).
    rho : array_like, shape (I, )
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
            r2d, rho[:, None] * r2d ** l * eval_legendre(l, t), k=spl_order
        )
        print("Moment n°{:2d} : {:+.10e}".format(l, m_l))
        

def radial_method(
    model: Model1D,
    rotation: RotationConfig,
    options: SolverOptions,
    output_options: OutputOptions,
) -> RadialResult:
    global G, J, L, spl_order
    global r1d, zeta
    global eval_phi_c, eval_omega

    start = time.perf_counter()

    if model.n_domains > 1:
        raise ValueError(
            "The radial solver only supports single-domain models."
        )

    r2d, t = initialize_mapping(
        model.r,
        options.angular_resolution,
    )

    num = initialize_radial_numerics(
        model.r,
        t,
        options,
    )

    G = model.G
    surface_pressure = model.surface_pressure
    radius = model.radius
    mass = model.mass

    r1d  = num.r1d.copy()
    zeta = num.r1d.copy()
    rho = model.rho.copy()

    I = num.n_points
    J = num.angular_resolution
    L = num.max_degree
    spl_order = num.spline_order

    eval_phi_c, eval_omega = configure_rotation_profile(
        rotation.profile,
        rotation.central_diff_rate,
        rotation.scale,
    )

    rotation_target = rotation.target
    full_rate = options.full_rate
    mapping_precision = options.mapping_precision
    max_iterations = options.max_iterations
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff, lub_l = find_phi_eff(r2d, rho, num)
    
    # Find pressure
    p = find_pressure(rho, dphi_eff, surface_pressure, num)
    
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
        omega = min(rotation_target, rotation_cap)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(
            r2d,
            rho,
            num,
            phi_eff=phi_eff,
            lub_l=lub_l,
        )

        # Find a new estimate for the mapping
        r2d, omega = find_new_mapping(
            omega,
            phi_g_l,
            dphi_g_l,
            phi_eff,
            num,
            eval_phi_c,
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
        pressure=p.copy(),

        effective_potential=phi_eff.copy(),
        effective_potential_derivative=dphi_eff.copy(),

        gravitational_potential_harmonics=phi_g_l.copy(),
        gravitational_potential_derivative_harmonics=dphi_g_l.copy(),

        mass=mass,
        radius=radius,

        rotation_target=rotation_target,
        rotation_rate=omega,

        polar_radius_history=np.asarray(polar_radius_history),
        iterations=iterations,
    )

    # Estimated error on Poisson's equation
    if output_options.plot.show_harmonics : 
        phi_g_harmonics(zeta, phi_g_l, radial=True)
    
    # Virial test
    if output_options.diagnostics.virial_test : 
        virial = Virial_theorem(r2d, rho, omega, phi_eff, p, verbose=True)   
    
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
        rota = eval_omega(r2d[:, (J-1)//2], 0.0, rotation_target)
        if output_options.model.dimensional : 
            r2d      *=               radius
            rho      *=     mass    / radius**3
            phi_eff  *= G * mass    / radius   
            dphi_eff *= G * mass    / radius**2
            p        *= G * mass**2 / radius**4
        write_model(
            output_options.model.filename,
            (I, J, mass, radius, rotation_target, G),
            r2d,
            additional_variables,
            zeta,
            p,
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
    mapping : array_like, shape (I, J)
        Model mapping.
    t : array_like, shape (J, )
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
        spline_order=spl_order,
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