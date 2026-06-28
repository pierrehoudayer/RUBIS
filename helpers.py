import numpy as np
from scipy.special  import roots_legendre

from rubis._utils   import DotDict
from rubis.domains  import find_domains

    
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
            if find_domains(radial_coordinate).n_domains > 1 :            
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