import numpy as np
from scipy.special  import roots_legendre

class DotDict(dict):  
    """
    Class that defines dictionaries with dot attributes.
    """   
    def __getattr__(*args):        
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val     
    __setattr__ = dict.__setitem__     
    __delattr__ = dict.__delitem__ 
    
def assign_method(method_choice, model_choice, radial_method, spheroidal_method) : 
    """
    Function assigning the method function to call to a given
    method_choice.

    Parameters
    ----------
    method_choice : string in {'auto', 'radial', 'spheroidal'}
        Method choice (cf. RUBIS.py)
    model_choice : string or DotDict instance
        Model choice (cf. RUBIS.py)
    radial_method : func 
        Function to call if model_choice is set to 'radial'
    spheroidal_method : func 
        Function to call if model_choice is set to 'spheroidal'

    Returns
    -------
    method_func : func in {radial_method, spheroidal_method}
        method function to call for the model deformation.
    """
    
    # Dealing with method_choice = 'auto'
    assert method_choice in {'auto', 'radial', 'spheroidal'}
    if method_choice == 'auto' :
        if isinstance(model_choice, DotDict) :       
            # Checking the number of domains in the composite polytrope 
            if len(np.atleast_1d(model_choice.indices)) > 1 : 
                method_choice = 'spheroidal'
            else : 
                method_choice = 'radial'
        else : 
            # Reading the file 
            radial_coordinate, *_ = np.genfromtxt(
                './Models/'+model_choice, skip_header=2, unpack=True
            )
            if find_domains(radial_coordinate).Nd > 1 :            
                method_choice = 'spheroidal'
            else : 
                method_choice = 'radial'
                
    # Assigning the adaquate method to method_choice
    if method_choice == 'radial' : 
        method_func = radial_method
    else : 
        method_func = spheroidal_method
    return method_func

def give_me_a_name(model_choice, rotation_target) : 
    """
    Constructs a name for the save file using the model name
    and the rotation target.

    Parameters
    ----------
    model_choice : string or Dotdict instance.
        File name or composite polytrope caracteristics.
    rotation_target : float
        Final rotation rate on the equator.

    Returns
    -------
    save_name : string
        Output file name.

    """
    radical = (
        'poly_|' + ''.join(
            str(np.round(index, 1))+"|" for index in np.atleast_1d(model_choice.indices)
        )
        if isinstance(model_choice, DotDict) 
        else model_choice.split('.txt')[0]
    )
    save_name = radical + '_deform_' + str(rotation_target) + '.txt'
    return save_name

def init_2D(r, M) :
    """
    Init function for the angular domain.

    Parameters
    ----------
    r : array_like, shape (N, ) 
        Radial coordinate from the 1D model.
    M : integer
        Angular resolution.

    Returns
    -------
    cth : array_like, shape (M, )
        Angular coordinate (equivalent to cos(theta)).
    map_n : array_like, shape (N, M)
        Isopotential mapping 
        (given by r(phi_eff, theta) = r for now).
    """
    map_n = np.tile(r, (M, 1)).T
    cth, _ = roots_legendre(M)
    return map_n, cth

def init_phi_c(rotation_profile, central_diff_rate, rotation_scale) : 
    """
    Defines the functions used to compute the centrifugal potential
    and the rotation profile with the adequate arguments.
    
    Parameters
    ----------
    rotation_profile : function(r, cth, omega, *args)
        Function used to compute the centrifugal potential, given 
        adequate additional arguments.
    central_diff_rate : float
        Parameter that may be used to compute the centrifugal potential
    rotation_scale : float
        Parameter that may be used to compute the centrifugal potential

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

def find_domains(var) :
    """
    Defines many tools to help the domain manipulation and navigation.
    
    Parameters
    ----------
    var : array_like, shape (Nvar, )
        Variable used to define the domains

    Returns
    -------
    dom : DotDict instance.
        Domains informations : {
            Nd : integer
                Number of domains.
            bounds : array_like, shape (Nd-1, )
                Zeta values at boundaries.
            interfaces : list of tuple
                Successives indices of domain interfaces
            beg, end : array_like, shape (Nd-1, ) of integer
                First (resp. last) domain indices.
            edges : array_like, shape (Nd+1, ) of integer
                All edge indices (corresponds to beg + origin + last).
            ranges : list of range()
                All domain index ranges.
            sizes : list of integers
                All domain sizes
            id : array_like, shape (Nvar, ) of integer
                Domain identification number. 
                /!\ if var is zeta, the Nvar = N+Ne!
            id_val : array_like, shape (Nd, ) of integer
                The id values.
            int, ext : array_like, shape (Nvar, ) of boolean
                Interior (resp. exterior, i.e. if rho = 0) domain.
            unq : array_like, shape (Nvar-(Nd-1), ) of integer
                Unique indices through the domains.
            }

    """
    dom = DotDict()
    Nvar = len(var)
    disc = True
    
    # Domain physical boundaries
    unq, unq_idx, unq_inv, unq_cnt = np.unique(
        np.round(var, 15), return_index=True, return_inverse=True, return_counts=True
    )
    cnt_mask = unq_cnt > 1
    dom.bounds = unq[cnt_mask]
    if len(dom.bounds) == 0 : disc = False
    
    # Domain interface indices
    cnt_idx, = np.nonzero(cnt_mask)
    idx_mask = np.in1d(unq_inv, cnt_idx)
    idx_idx, = np.nonzero(idx_mask)
    srt_idx  = np.argsort(unq_inv[idx_mask])
    dom.interfaces = np.split(
        idx_idx[srt_idx], np.cumsum(unq_cnt[cnt_mask])[:-1]
    )
    if disc : dom.end, dom.beg = np.array(dom.interfaces).T
    
    # Domain ranges and sizes
    dom.unq    = unq_idx
    dom.Nd     = len(dom.bounds) + 1
    if disc :
        dom.edges  = np.array((0, ) + tuple(dom.beg) + (Nvar, ))
    else :
        dom.edges  = np.array((0, Nvar, ))
    dom.ranges = list(map(range, dom.edges[:-1], dom.edges[1:]))
    dom.sizes  = list(map(len, dom.ranges))

    # Domain indentification
    dom.id      = np.hstack([d*np.ones(S) for d, S in enumerate(dom.sizes)])
    dom.id_val  = np.unique(dom.id)
    dom.ext     = dom.id == dom.Nd - 1
    dom.int     = np.invert(dom.ext)
    dom.unq_int = np.unique(var[dom.int], return_index=True)[1]
    
    return dom

def valid_reciprocal_domain(x, df, safety=1e-4) :
    """
    Find the valid f domain for a reciprocal function interpolation (i.e. 
    of the function x(f)) knowing df/dx. The function f is allowed to have
    another variable y, in which case the valid domain have the same shape
    as f and is estimated for each value of y.
    
    Parameters
    ----------
    x : array_like, shape (N, )
        Variable along which the f-derivative is taken
    df : array_like, shape (N, ) or shape (N, M)
        Derivative of f with respect to f (the partial derivative w.r.t x should
        correspond to the first axis).

    Returns
    -------
    valid : array_like of boolean, shape (N, ) or shape (N, M)
        Valid domain for the reciprocal function interpolation.
    """
    df = np.atleast_2d(df.T).T
    valid = np.ones_like(df, dtype='bool')
    idx = np.arange(len(x))
    for k, dpk in enumerate(df.T) :
        idx_max = len(idx)
        condition = (dpk < safety) & (x > safety)
        if np.any(condition) : idx_max = np.min(np.argwhere(condition))
        valid[:, k] = (idx < idx_max) & (x > safety)
    valid = np.squeeze(valid)
    return valid
    
def write_model(fname, params, map_n, additional_var, *args) : 
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
    params : tuple
        Model parameters to write on the first line.
    map_n : array_like, shape (N, M)
        level surfaces mapping.
    additional_var : tuple of arrays
        Tuple of (unchanged) additional variables.
    args : tuple with N_args elements
        Variables to be saved in addition to map_n.

    """
    header = "".join(str(c)+" " for c in params)[:-1]
    np.savetxt(
        'Models/'+fname, np.hstack((map_n, np.vstack(args + (*additional_var,)).T)), 
        header=header,  comments=''
    )