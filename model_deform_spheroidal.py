import time
import numpy             as np
import scipy.sparse      as sps
import scipy.special     as sp
from gc                  import collect
from scipy.interpolate   import CubicHermiteSpline
from scipy.linalg.lapack import dgbsv
from scipy.special       import roots_legendre, eval_legendre

from legendre            import find_r_eq, find_r_pol, pl_eval_2D, pl_project_2D, Legendre_coupling
from numerical           import integrate, integrate2D, interpolate_func, lagrange_matrix_P
from polytrope           import composite_polytrope
from helpers             import DotDict, find_domains, plot_f_map, phi_g_harmonics

def init_1D() : 
    """
    Function reading the 1D model file 'MOD_1D' (or generating a
    polytrope if MOD_1D is a dictionary). If additional variables are 
    found in the file, they are left unchanged and returned in the 
    'SAVE' file.

    Returns
    -------
    P0 : float
        Value of the surface pressure after normalisation.
    N : integer
        Radial resolution of the model.
    M : float
        Total mass of the model.
    R : float
        Radius of the model.
    r : array_like, shape (N, ), [GLOBAL VARIABLE]
        Radial coordinate after normalisation.
    zeta : array_like, shape (N+NE, ), [GLOBAL VARIABLE]
        Spheroidal coordinate
    rho : array_like, shape (N, )
        Radial density of the model after normalisation.
    other_var : array_like, shape (N, N_var)
        Additional variables found in 'MOD1D'.

    """
    if isinstance(model_choice, DotDict) :        
        # The model properties are user-defined
        N = model_choice.res    or 1001
        M = model_choice.mass   or 1.0
        R = model_choice.radius or 1.0
        
        # Polytrope computation
        model = composite_polytrope(model_choice)
        
        # Normalisation
        r   = model.r     /  R
        rho = model.rho   / (M/R**3)
        other_var = np.empty_like(r)
        P0  = model.p[-1] / (G*M**2/R**4)
        
    else : 
        # Reading file 
        surface_pressure, radial_res = np.genfromtxt(
            './Models/'+model_choice, max_rows=2, unpack=True
        )
        r1D, rho1D, *other_var = np.genfromtxt(
            './Models/'+model_choice, skip_header=2, unpack=True
        )
        N = int(radial_res)
        
        # Normalisation
        R = r1D[-1]
        dom = find_domains(r1D)
        M = 4*np.pi * sum(
            integrate(x=r1D[D], y=r1D[D]**2 * rho1D[D]) for D in dom.ranges
        )
        r   = r1D / (R)
        rho = rho1D / (M/R**3)
        P0  = surface_pressure / (M**2/R**4)
        
        # We assume that P0 is already normalised by G if R ~ 1 ...
        if not np.allclose(R, 1) :
            P0 /= G
    
    # Spheroidal coordinate
    re = np.linspace(1, 2, NE)
    zeta = np.hstack((r, 1+sp.betainc(2, 2, re-1)))
    
    return P0, N, M, R, r, zeta, rho, other_var
        
def init_2D() :
    """
    Init function for the angular domain.

    Parameters
    ----------
    None.

    Returns
    -------
    cth : array_like, shape (M, ), [GLOBAL VARIABLE]
        Angular coordinate (equivalent to cos(theta)). 
    weights : array_like, shape (M, ), [GLOBAL VARIABLE]
        Angular weights for the Legendre quadrature.
    map_n : array_like, shape (N, M)
        Isopotential mapping 
        (given by r(phi_eff, theta) = r for now).
    """
    cth, weights = roots_legendre(M)
    map_n = np.tile(r, (M, 1)).T
    return cth, weights, map_n   

def init_phi_c() : 
    """
    Defines the functions used to compute the centrifugal potential
    and the rotation profile with the adequate arguments.

    Returns
    -------
    phi_c : function(r, cth, omega)
        Centrifugal potential
    w : function(r, cth, omega)
        Rotation profile

    """
    nb_args = (
          rotation_profile.__code__.co_argcount 
        - len(rotation_profile.__defaults__ or '')
    )
    mask = np.array([0, 1]) < nb_args - 3
    
    # Creation of the centrifugal potential function
    args_phi = np.array([central_diff_rate, rotation_scale])[mask]
    phi_c = lambda r, cth, omega : rotation_profile(r, cth, omega, *args_phi)
    
    # Creation of the rotation profile function
    args_w = np.hstack((np.atleast_1d(args_phi), (True,)))
    w = lambda r, cth, omega : rotation_profile(r, cth, omega, *args_w)
    return phi_c, w

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
    for D in dom.ranges : 
        
        # Find the lagrange matrices per domain
        lag_mat = lagrange_matrix_P(zeta[D], order=KLAG)
        
        # Define sparse matrices 
        Lsp.append(sps.dia_matrix(lag_mat[..., 0]))
        Dsp.append(sps.dia_matrix(lag_mat[..., 1]))
    
    return Lsp, Dsp


def find_gravitational_moments(map_n, rho_n, max_degree=14) :
    """
    Find the gravitational moments up to max_degree.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.
    rho_n : array_like, shape (N, )
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
            map_n, rho_n[:, None] * map_n ** l * eval_legendre(l, cth), 
            domains=dom.ranges[:-1], k=KSPL
        )
        print("Moment n°{:2d} : {:+.10e}".format(l, m_l))


def find_pressure(rho, dphi_eff) :
    """
    Find the pressure evaluated on zeta thanks to the hydrostatic
    equilibrium.

    Parameters
    ----------
    rho : array_like, shape (N, )
        Density profile.
    dphi_eff : array_like, shape (N, )
        Effective potential derivative with respect to zeta.

    Returns
    -------
    P : array_like, shape (N, )
        Pressure profile.

    """
    dP = - rho * dphi_eff[dom.int]        
    P  = interpolate_func(
        zeta[dom.unq_int], dP[dom.unq_int], der=-1, k=KSPL, prim_cond=(-1, P0)
    )(zeta[dom.int])
    return P


def find_metric_terms(map_n) : 
    """
    Finds the metric terms, i.e the derivatives of r(z, t) 
    with respect to z or t (with z := zeta and t := cos(theta)).

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Isopotential mapping.

    Returns
    -------
    dr : DotDict instance
        The mapping derivatives : {
            _   = r(z, t),
            t   = r_t(z, t),
            tt  = r_tt(z, t),
            z   = r_z(z, t),
            zt  = r_zt(z, t),
            ztt = r_ztt(z, t)
            }          
    """
    dr = DotDict()
    dr._ = map_n
    map_l = pl_project_2D(dr._, L)
    _, dr.t, dr.tt = pl_eval_2D(map_l, cth, der=2)
    dr.z = np.array(
        [np.hstack(
            [interpolate_func(zeta[D], rk[D], der=1, k=KSPL)(zeta[D]) 
             for D in dom.ranges[:-1]]
        ) for rk in map_n.T]            # <- mapping derivative potentially
    ).T                                 #    discontinous on the interfaces
    map_l_z = pl_project_2D(dr.z, L)
    _, dr.zt, dr.ztt = pl_eval_2D(map_l_z, cth, der=2)
    return dr

def find_external_mapping(dr) : 
    """
    Complete the internal mapping (0 <= z <= 1) to an external 
    domain (1 <= z <= 2). By convention, this continuation reaches 
    the spherical metric at z=2, thus making it handy to impose 
    the boundary conditions on this point. Please note that the
    mapping thus defined is not derivable at the interface between 
    the internal and external domains (hence in z = 1).

    Parameters
    ----------
    dr : DotDict instance
        The internal mapping and its derivatives with respect to z and t

    Returns
    -------
    dr : DotDict instance
        The commplete mapping from 0 <= z <= 2.
    """
    # External zeta variable 
    z = zeta[dom.ext].reshape((-1, 1))
    
    # Internal mapping constraints
    surf, dsurf, ddsurf = dr._[-1], dr.t[-1], dr.tt[-1]
    
    # External mapping and derivatives
    max_degree = 3
    dr._  = np.vstack((dr._, z - (1-surf)*(2-z)**max_degree))
    dr.t  = np.vstack((dr.t,       dsurf *(2-z)**max_degree))
    dr.tt = np.vstack((dr.tt,     ddsurf *(2-z)**max_degree))
    dr.z  = np.vstack((dr.z, 1 + (1-surf)*(2-z)**(max_degree-1)*max_degree))
    
    # Sanity check
    assert not np.any(dr.z < 0)
    
    return dr

def find_all_couplings(dr, alpha=2) :
    """
    Find all the couplings needed to solve Poisson's equation in 
    spheroidal coordinates.

    Parameters
    ----------
    dr : DotDict instance
        The mapping and its derivatives with respect to z and t
    alpha : float, optional
        Constant to be either set to 1 (typically for divergences)
        or 2 (Laplacians).

    Returns
    -------
    Pll : DotDict instance
        Harmonic couplings. They are caracterised by their related 
        metric term : {
            zz : array_like, shape (N+NE, Nl, Nl)
                coupling associated to phi_zz,
            zt : array_like, shape (N+NE, Nl, Nl)
                            //         phi_zt,
            tt : array_like, shape (N+NE, Nl, Nl)
                            //         phi_tt,
            BC : array_like, shape (N+NE, Nl, Nl)
                coupling used to ensure the gradient continuity.
        }

    """
    Pll = DotDict()
    l = np.arange(0, L, 2)
    
    Pll.zz = Legendre_coupling(
        (dr._**2 + (1-cth**2) * dr.t**2) / dr.z, L, der=(0, 0)
    )
    Pll.zt = Legendre_coupling(
        (1-cth**2) * dr.tt - 2*cth * dr.t, L, der=(0, 0)
    ) + alpha * Legendre_coupling(
        (1-cth**2) * dr.t, L, der=(0, 1)
    )
    Pll.tt = Legendre_coupling(dr.z, L, der=(0, 0)) * l*(l+1)
    
    Pll.BC = Legendre_coupling(1/dr.z, L, der=(0, 0))
    
    return Pll 

def find_Poisson_coefs(kl, ku, Pll, rhs_l, rescale) :
    """
    Finds the coefficients to fill the Poisson matrix (thanks
    to Pll) and the right-hand side (using rhs).

    Parameters
    ----------
    ku : integer
        Number of upper band in the matrix.
    kl : integer
        Number of lower band in the matrix.
    Pll : DotDict instance
        Harmonic couplings.
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
    
    for d, D in enumerate(dom.ranges) : 
        
        # Domain properties
        beg_i, end_i = (2*dom.edges[d]+1)*Nl, (2*dom.edges[d+1]-1)*Nl
        beg_j, end_j = (2*dom.edges[d]+0)*Nl, (2*dom.edges[d+1]-0)*Nl
        Lsp_d_broad = Lsp[d].data[::-1, :, None, None]
        Dsp_d_broad = Dsp[d].data[::-1, :, None, None]
        size = dom.sizes[d]
        
        # Vector filling
        if d < dom.Nd - 1 : 
            b[beg_i:end_i:2] = (Lsp[d] @ rhs_l[D]).flatten()
            
        # Main matrix parts filling
        temp = np.empty((2*KLAG, size, 2*Nl, 2*Nl))
        temp[..., 0::2, 0::2] = (
            + Dsp_d_broad * Pll.zz[D]
            - Lsp_d_broad * Pll.zt[D]
        )
        temp[..., 0::2, 1::2] = - Lsp_d_broad * Pll.tt[D]
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
            coefs[ku-3*Nl+1+0:ku-Nl+1:2, beg_j+0:beg_j+2*Nl:2] = -Pll.BC[D[0]]
            coefs[ku-3*Nl+1+1:ku-Nl+1:2, beg_j+1:beg_j+2*Nl:2] = -np.eye(Nl)
        
        # Outer boundary conditions
        if d == dom.Nd - 1 : 
            coefs[ku-Nl+1:ku+1, -2*Nl+0::2] = np.eye(Nl)
            coefs[ku-Nl+1:ku+1, -2*Nl+1::2] = np.diag((l+1)/2)
        else :  
            coefs[ku-Nl+1+0:ku+Nl+1:2, end_j-2*Nl+0:end_j:2] = Pll.BC[D[-1]]
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


def find_phi_eff(map_n, rho_n, phi_eff=None) :
    """
    Determination of the effective potential from a given mapping
    (map_n, which gives the lines of constant density), and a given 
    rotation rate (omega_n). This potential is determined by solving
    the Poisson's equation on each degree of the harmonic decomposition
    (giving the gravitational potential harmonics which are also
    returned) and then adding the centrifugal potential.

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
    rho_n : array_like, shape (N, )
        Current density on each equipotential.
    phi_eff : array_like, shape (N, ), optional
        If given, the current effective potential on each 
        equipotential. If not given, it will be calculated inside
        this fonction. The default is None.

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
    # Metric terms and coupling integral computation
    dr     = find_metric_terms(map_n)
    r2rz_l = pl_project_2D(dr._**2 * dr.z, L) / (np.arange(L) + 1/2)
    rhs_l  = 4*np.pi * rho_n[:, None] * r2rz_l[:, ::2]
    dr     = find_external_mapping(dr)
    Pll    = find_all_couplings(dr, alpha=2)
    
    # Vector and band matrix storage characteristics
    Nl = (L+1)//2
    kl = (2*KLAG + 1) * Nl - 1
    ku = (2*KLAG + 1) * Nl - 1
    
    # Determination of matrix blocs (and b) from coupling harmonics (and rhs harmonics)
    # Rescale the coefficients in both coefs and b if AB_SCALE==True.
    coefs, b, col_scale = find_Poisson_coefs(kl, ku, Pll, rhs_l, rescale=rescale_ab)  
    
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


def find_new_mapping(map_n, omega_n, phi_g_l, dphi_g_l, phi_eff, dphi_eff) :
    """
    Find the new mapping by comparing the effective potential
    and the total potential (calculated from phi_g_l and omega_n).

    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Current mapping.
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
    dr = find_metric_terms(map_n)
    dr = find_external_mapping(dr)
    
    # 2D gravitational potential
    eq = (M-1)//2
    up = np.arange(eq+1)
    phi2D_g = pl_eval_2D( phi_g_l, cth[up])
    dphi2D_g_dz = pl_eval_2D(dphi_g_l, cth[up])
    dphi2D_g = dphi2D_g_dz / dr.z[:, up]
        
    ### Find the adaptive rotation rate
    valid_z = zeta > 0.5
    valid_r = dr._[valid_z, eq]
    phi1D_c, dphi1D_c = eval_phi_c(valid_r, 0.0, omega_n) / valid_r ** 3
    dphi1D_c -= 3 * phi1D_c / valid_r
    phi1D  =  phi2D_g[valid_z, eq] +  phi1D_c
    dphi1D = dphi2D_g[valid_z, eq] + dphi1D_c
    
    dom_r = find_domains(valid_r)
    r_est = CubicHermiteSpline(
        x=phi1D[dom_r.unq], y=valid_r[dom_r.unq], dydx=dphi1D[dom_r.unq] ** -1
    )(targets[-1])
    omega_n_new = omega_n * r_est**(-1.5)
                
    ### Find the new mapping using the reciprocal interpolation
    # Define the adaptive mesh
    new_res = int(1.0/(np.finfo(float).eps)**0.2)
    d2phi_eff = np.hstack((0.0, np.abs(np.diff(dphi_eff[dom.unq]))))
    z_new = interpolate_func(d2phi_eff.cumsum(), zeta[dom.unq], k=1)(
        np.linspace(0.0, d2phi_eff.sum(), new_res)
    )
    z_new = 2 * (z_new - z_new[0]) / (z_new[-1] - z_new[0])
    
    # Cubic Hermite splines
    r_splines = [CubicHermiteSpline(
        x=zeta[dom.unq], y=dr._[dom.unq, k], dydx=dr.z[dom.unq, k]
    ) for k in up]
    p_splines = [CubicHermiteSpline(
        x=zeta[dom.unq], y=phi2D_g[dom.unq, k], dydx=dphi2D_g_dz[dom.unq, k]
    ) for k in up]
    
    # Interpolated variables
    r_ipl, dr_ipl = [
        np.array([r_spl(z_new, nu=nu) for r_spl in r_splines]).T for nu in [0, 1]
    ]
    phi_g_ipl, dphi_g_ipl = [
        np.array([p_spl(z_new, nu=nu) for p_spl in p_splines]).T for nu in [0, 1]
    ]
    phi_c_ipl, dphi_c_ipl = np.moveaxis(np.array([
        eval_phi_c(rk , ck, omega_n_new) for rk, ck in zip(r_ipl[:, up].T, cth[up])
    ]), 0, 2)
    phi_ipl  =  phi_c_ipl +  phi_g_ipl
    dphi_ipl = dphi_c_ipl + dphi_g_ipl / dr_ipl
    
    # Finding the valid interpolation domain
    valid = np.ones_like(phi_ipl, dtype='bool')
    idx = np.arange(len(z_new))
    safety = 1e-4
    for k, dpk in enumerate(dphi_ipl.T) :
        idx_max = len(z_new)
        if np.any((dpk < safety)&(z_new > safety)) :
            idx_max = np.min(np.argwhere((dpk < safety)&(z_new > safety)))
        valid[:, k] = (idx < idx_max) & (z_new > safety)
        
    # Estimate at target values
    map_est = np.zeros_like(map_n[:, up])
    map_est[1:] = np.array([
        CubicHermiteSpline(x=pk[vk], y=rk[vk], dydx=dpk[vk]**-1)(targets[1:]) 
        for rk, pk, dpk, vk in zip(r_ipl.T, phi_ipl.T, dphi_ipl.T, valid.T)
    ]).T
    map_est[dom.beg[:-1]] = map_est[dom.end[:-1]]
        
    ### New mapping
    map_n_new = np.hstack((map_est, np.flip(map_est, axis=1)[:, 1:]))
        
    return map_n_new, omega_n_new


def Virial_theorem(map_n, rho_n, omega_n, phi_g_l, P, verbose=False) : 
    """
    Compute the Virial equation and gives the resukt as a diagnostic
    for how well the hydrostatic equilibrium is satisfied (the closer
    to zero, the better).
    
    Parameters
    ----------
    map_n : array_like, shape (N, M)
        Mapping
    rho_n : array_like, shape (N, )
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
    volumic_potential_energy = lambda rk, ck, D : -rho_n[D] * pl_eval_2D(phi_g_l[D], ck)
    potential_energy = integrate2D(
        map_n, volumic_potential_energy, domains=dom.ranges[:-1], k=KSPL
    )
    
    # Kinetic energy
    volumic_kinetic_energy = lambda rk, ck, D : (  
       0.5 * rho_n[D] * (1-ck**2) * rk[D]**2 * eval_w(rk[D], ck, omega_n)**2
    )
    kinetic_energy = integrate2D(
        map_n, volumic_kinetic_energy, domains=dom.ranges[:-1], k=KSPL
    )
    
    # Internal energy
    internal_energy = integrate2D(map_n, P, domains=dom.ranges[:-1], k=KSPL)
    
    # Surface term
    surface_term = 2*np.pi * (map_n[-1]**3 @ weights) * P[-1]
    
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

    
def write_model(fname, map_n, *args) : 
    """
    Saves the deformed model in the file named fname. The resulting 
    table has dimension (N, M+N_args+N_var) where the last N_var columns
    contains the additional variables given by the user (the lattest
    are left unchanged during the whole deformation). The dimensions N & M,
    as well as the global paramaters mass, radius, ROT, G
    are written on the first line.

    Parameters
    ----------
    fname : string
        File name
    map_n : array_like, shape (N, M)
        level surfaces mapping.
    args : TUPLE with N_args elements
        Variables to be saved in addition to map_n.

    """
    np.savetxt(
        'Models/'+fname, np.hstack((map_n, np.vstack(args + (*additional_var,)).T)), 
        header=f"{N} {M} {mass} {radius} {rotation_target} {G}", 
        comments=''
    )       

def spheroidal_method(*params) : 
    """
    Main routine for the centrifugal deformation method in spheroidal coordinates.

    Parameters
    ----------
    params : tuple
        All method parameters. Please refer to the documentation in RUBIS.py

    """
    
    # Global parameters
    global model_choice, rotation_profile, rotation_target, central_diff_rate, \
    rotation_scale, L, M, full_rate, mapping_precision, KSPL, KLAG, output_params, \
    NE, rescale_ab
    model_choice, rotation_profile, rotation_target, central_diff_rate, \
    rotation_scale, L, M, full_rate, mapping_precision, KSPL, KLAG, output_params, \
    NE, rescale_ab = params
    
    # Global constants, variables and functions
    global G, P0, N, mass, radius, r, zeta, rho_n, additional_var, \
    cth, weights, eval_phi_c, eval_w, Lsp, Dsp, dom
    G = 6.67384e-8  # <- Gravitational constant
    
    start = time.perf_counter()
    
    # Definition of the 1D-model
    P0, N, mass, radius, r, zeta, rho_n, additional_var = init_1D() 
    
    # Domains identification
    dom = find_domains(zeta)
    
    # Angular domain preparation
    cth, weights, map_n = init_2D()
    
    # Centrifugal potential definition
    eval_phi_c, eval_w = init_phi_c()
    
    # Find the lagrange matrices per domain
    Lsp, Dsp = init_sparse_matrices_per_domain()
    
    # Initialisation for the effective potential
    phi_g_l, dphi_g_l, phi_eff, dphi_eff = find_phi_eff(map_n, rho_n)
    
    # Find pressure
    P = find_pressure(rho_n, dphi_eff)
    
    # Iterative centrifugal deformation
    r_pol = [0.0, find_r_pol(map_n, L)]
    n = 0
    print(
        "\n+---------------------+",
        "\n| Deformation started |", 
        "\n+---------------------+\n"
    )
    
    while abs(r_pol[-1] - r_pol[-2]) > mapping_precision :
        
        # Current rotation rate
        omega_n = min(rotation_target, ((n+1)/full_rate) * rotation_target)
        
        # Effective potential computation
        phi_g_l, dphi_g_l, phi_eff = find_phi_eff(map_n, rho_n, phi_eff)
        
        # Update the mapping
        map_n, omega_n = find_new_mapping(
            map_n, omega_n, phi_g_l, dphi_g_l, phi_eff, dphi_eff
        )        

        # Renormalisation
        r_corr    = find_r_eq(map_n, L)
        m_corr    = integrate2D(map_n, rho_n, domains=dom.ranges[:-1])   
        radius   *= r_corr
        mass     *= m_corr
        map_n    /=             r_corr
        rho_n    /= m_corr    / r_corr**3
        phi_eff  /= m_corr    / r_corr
        dphi_eff /= m_corr    / r_corr    # <- /!\ This is a derivative w.r.t. to zeta
        P        /= m_corr**2 / r_corr**4
        
        # Update the polar radius
        r_pol.append(find_r_pol(map_n, L))
        
        # Iteration count
        n += 1
        DEC = int(-np.log10(mapping_precision))
        print(f"Iteration n°{n:02d}, R_pol = {r_pol[-1].round(DEC)}")
        
    finish = time.perf_counter()
    print(
        "\n+------------------+",
        "\n| Deformation done |", 
        "\n+------------------+\n"
    )
    print(f'Time taken: {round(finish-start, 2)} secs')  
    
    # Estimated error on Poisson's equation
    dr = find_metric_terms(map_n)
    dr = find_external_mapping(dr)
    if output_params.show_harmonics :
        phi_g_harmonics(r, phi_g_l, r_pol[-1], dr=dr, show=True)
    
    # Virial test
    if output_params.virial_test :
        virial = Virial_theorem(map_n, rho_n, omega_n, phi_g_l, P, verbose=True)   
    
    # Plot model
    if output_params.show_model :
        plot_f_map(
            map_n, np.log10(rho_n+rho_n.max()**-1), phi_eff, L, 
            angular_res=output_params.plot_resolution,
            cmap='cividis', 
            disc=dom.end[:-1],
            label=r"$\rho \times {\left(M/R_{\mathrm{eq}}^3\right)}^{-1}$"
        )
    
    # Gravitational moments
    if output_params.gravitational_moments :
        find_gravitational_moments(map_n, rho_n)
    
    # Model writing
    if output_params.save_model :
        map_n    *=               radius
        rho_n    *=     mass    / radius**3
        phi_eff  *= G * mass    / radius   
        dphi_eff *= G * mass    / radius**2
        P        *= G * mass**2 / radius**4
        write_model(output_params.save_name, map_n, r, P, rho_n, phi_eff)
    
        
    
    
    
    

    