import numpy as np

from helpers                 import DotDict, give_me_a_name, assign_method
from plot                    import get_cmap_from_proplot
from rotation_profiles       import *
from model_deform_radial     import radial_method
from model_deform_spheroidal import spheroidal_method

def set_params() : 
    """
    Function documenting and returning the RUBIS interface parameters.

    Returns
    -------
    method_choice : string
        Name of the method to be used for the model deformation. 
        Options are {'auto', 'radial', 'spheroidal'}. The 'radial'
        and 'spheroidal' options will respectively call the 
        model_deform_radial() and model_deform_spheroidal() methods, while
        the 'auto' option will automatically decide which one to choose
        depending on the presence of discontinuities or not in the 
        1D model given by 'model_choice' (cf. below). The latter is 
        recommended unless performing specific tests.
        
    model_choice : string or DotDict instance
        Name of the file containing the 1D model or dictionary containing
        the information requiered to compute a composite polytrope with
        caracteristics: {
            indices : array_like, shape(D, )
                Each region polytropic index
            target_pressures : array_like, shape(D, )
                Normalised interface pressure values (surface included).
            density_jumps : array_like, shape(D-1, )
                Density ratios above and below each interface (surface excluded).
                The default value is None
            R : float, optional
                Composite polytrope radius. The default value is 1.0
            M : float, optional
                Composite polytrope mass. The default value is 1.0
            res : int, optional
                Number of points. The default value is 1001
        }
        Please refer to the composite_polytrope() documentation for more information.
        
    rotation_profile : function(r, cth, omega)
        Function used to compute the centrifugal potential and its 
        derivative. Possible choices are {solid, lorentzian, plateau}.
        Explanations regarding this profiles are available in the 
        corresponding functions in the rotation_profiles.py file.
    rotation_target : float
        Target for the rotation rate, expressed in units of the Keplerian
        rotation rate. The method might still converge for values slightly above
        1.0 because the Keplerian rotation rate is not exactly the critical
        rotation rate. 
    rate_difference : float
        The rotation rate difference between the centre and equator in the
        cylindrical rotation profile. For instance, rate_difference = 0.0
        would corresond to a solid rotation profile while 
        rate_difference = 0.5 indicates that the star's centre rotates 50%
        faster than the equator.
        Only appears in cylindrical rotation profiles.
    rotation_scale : float
        Homotesy factor on the x = r*sth / Req axis for the rotation profile.
        Only appear in the plateau rotation profile, where it gives an 
        indication for the lenght of the "plateau".
        
    max_degree : integer, odd
        Maximum l degree to be considered in order to do the
        harmonic projection. Note that, because of the equatorial symmetry,
        the actual number of harmonic used is (L+1)//2 (if odd) or L//2+1 
        (if even), i.e. the number of even harmonics. Curiously, though 
        the degrees used for the deformation are even, it is strongly 
        recommended to choose it as an odd number and, even better, as
        being exactly equal at 'angular_resolution' (see below).  
    angular_resolution : integer, odd
        Angular resolution used for the 2D mapping definition. Best to 
        take an odd number in order to include the equatorial radius.
    full_rate : integer
        Number of iterations before reaching the 'rotation_target', using
        linear increments. Considering that an adaptive rotation rate 
        is computed to ensure that (in most cases) the rotation rate does 
        not exceed a critical value, this number can often be set to one.
        For rotation rates very close to the critical value, or for very
        differential rotation profiles (or both at the same time), one 
        may benefit setting it to 3 or up to 5. It is nevertheless rarely
        beneficial to set it to a value beyond this one.
    mapping_precision : float
        Convergence criterion on succesive polar radii to . Values 
        recommended are {1e-10, 1e-11, 1e-12}, but it is rarely beneficial to 
        set it below 1e-12 (it mostly slows down the process by waiting for a 
        lucky radii difference). Some particularly stiff deformations might
        not converge beyond 1e-8, however.
    lagrange_order : integer
        Rather specific parameter giving the order of the finite difference
        scheme in Poisson's equation. The effective order should normally be of
        2*lagrange_order-1 but it reaches 2*lagrange_order in RUBIS thanks to
        a Poisson's equation solving on a "superconvergence grid". 
        It is recommended setting it to 2 but deformations using the 
        model_deform_radial() method may benefit from a value of 3 if the model
        radial resolution of ~1000 or below. Values above 3 only increase the
        numerical error, however.
    spline_order : integer
        Choice of B-spline order in integration / interpolation
        routines. Please note that many interpolations in RUBIS use cubic
        Hermite splines, which degree is fixed to 3. This option will thus
        have only limited impact on the results. Nevertheless, 5 is 
        recommanded (must be in {1, 3, 5} in anycase).
        
    output_params : DotDict instance
        Dictionary containing various parameters responsible for the output : {
            show_harmonics : boolean
                Whether to show the gravitational potential harmonics at the 
                end of the deformation.
            virial_test : boolean
                Whether to perform the Virial test, i.e. verifying how close 
                the Virial theorem is verified.
            show_model : boolean
                Whether to show the deformed model at the end of the procedure.
            plot_resolution : integer
                Angular resolution used when displaying 2D variables with 
                plot_f_map(). It is again recommended to choose an odd number.
            plot_surfaces : boolean
                Whether to display the isopotential surfaces on the right side of
                the plot.
            plot_cmap_f : string
                Colormap used when displaying the f function (if show_model is 
                True).
            plot_cmap_surfaces : string
                Colormap used when displaying the isopotentials (if plot_surfaces 
                is True).
            gravitational_moments : boolean
                Whether to compute the gravitational moments of the deformed 
                model.
            radiative_flux : boolean
                Whether to compute the radiative flux at the surface of the model
                by estimating the gravity darkening.
            plot_flux_lines : boolean
                Should the flux lines be added on top of the model plot?
            flux_origin : float
                Zeta value on which the flux will be assumed to be constant. Because the
                metric terms are not all defined on the origin, this value must be greater
                than zero (but can eventualy chosen to be quite small).
            flux_lines_number : integer
                Number of flux lines to be computed. High values will tend to increase
                the precision on the surface flux, but considerably extend the computation
                time. Values ~ 30 usually gives an honorable precision for not overly 
                complex surfaces.
            show_T_eff : boolean
                Whether to show the effective temperature instead of the radiative
                flux amplitude on the 3D surface.
            flux_res : tuple of floats (res_t, res_p)
                Gives the resolution of the 3D surface in theta and phi coordinates 
                respectively.
            flux_cmap : ColorMap instance
                Colormap used to display the surface radiative flux.
            dim_model : boolean
                Whether to redimension the model or not.
            save_model : boolean
                Whether to save the deformed model. 
            save_name : string
                Filename in which the deformed model will be saved 
                (if save_model = True).  This tedious task can be avoided by 
                using the give_me_a_name(model_choice, rotation_target) function 
                instead.
        }
        
    external_domain_res : integer
        Radial resolution for the external domain. Low values enhance the
        performances but too low values might cause instabilities. 
        201 is generally sufficient, but values below that might be enough
        in specific contexts.
    rescale_ab : boolean
        Whether to rescale the Poisson's matrix before calling 
        LAPACK's LU decomposition routine. Seems to have a 
        negligible impact on precision in most cases but it is 
        recommended to set it as True, as it only slightly alter performance.
    """
    
    #### METHOD CHOICE ####
    method_choice = 'auto'
    
    #### MODEL CHOICE ####
    model_choice = DotDict(indices = 3.0, target_pressures = -np.inf)
    # model_choice = DotDict(
    #     indices = (2.0, 1.0, 3.0, 1.5, 2.0, 4.0), 
    #     target_pressures = (-1.0, -2.0, -3.0, -5.0, -7.0, -np.inf), 
    #     density_jumps = (0.3, 0.2, 2.0, 0.5, 0.2)
    # )
    # model_choice = 'Jupiter.txt'

    #### ROTATION PARAMETERS ####      
    rotation_profile = solid
    rotation_target = 0.9
    central_diff_rate = 1.0
    rotation_scale = 1.0
    
    #### SOLVER PARAMETERS ####
    max_degree = angular_resolution = 201
    full_rate = 1
    mapping_precision = 1e-10
    lagrange_order = 3
    spline_order = 5
    
    #### OUTPUT PARAMETERS ####
    output_params = DotDict(
        show_harmonics = False,
        virial_test = True,
        show_model = True,
        plot_resolution = 501,
        plot_surfaces = True,
        plot_cmap_f = get_cmap_from_proplot("Greens1_r"),
        plot_cmap_surfaces = get_cmap_from_proplot("Greys"),
        gravitational_moments = False,
        radiative_flux = True,
        plot_flux_lines = True,
        flux_origin = 0.05,
        flux_lines_number = 20,
        show_T_eff = True,
        flux_res = (200, 100),
        flux_cmap = get_cmap_from_proplot("Stellar_r"),
        dim_model = False,
        save_model = False,
        save_name = give_me_a_name(model_choice, rotation_target)
    )
    
    #### SPHEROIDAL PARAMETERS ####
    external_domain_res = 201
    rescale_ab = True
    
    return (
        method_choice,
        model_choice, 
        rotation_profile, rotation_target, central_diff_rate, rotation_scale, 
        max_degree, angular_resolution, full_rate,
        mapping_precision, spline_order, lagrange_order,
        output_params, 
        external_domain_res, rescale_ab
    )
    
if __name__ == '__main__' :
    
    # Setting the global parameters
    method_choice, model_choice, rotation_profile, rotation_target,     \
    central_diff_rate, rotation_scale, max_degree, angular_resolution,  \
    full_rate, mapping_precision, spline_order, lagrange_order,         \
    output_params, external_domain_res, rescale_ab = set_params()
    
    # Choosing the method to call
    method_func = assign_method(
        method_choice, model_choice, radial_method, spheroidal_method
    )
    
    # Performing the deformation
    method_func(
        model_choice, 
        rotation_profile, rotation_target, central_diff_rate, rotation_scale, 
        max_degree, angular_resolution, full_rate,
        mapping_precision, spline_order, lagrange_order,
        output_params, 
        external_domain_res, rescale_ab
    )
