import numpy as np
from pathlib import Path

from rubis.options           import (
    OutputOptions,
    DiagnosticOptions,
    PlotOptions,
    RadiativeFluxOptions,
    ModelOutputOptions,
)
from rubis.models            import CompositePolytropeConfig
from rubis.domains           import find_domains
from rubis.rotation_profiles import solid, lorentzian, plateau
from rubis.solvers           import radial_method, spheroidal_method


def set_params():
    """Return the editable parameters used by the RUBIS script."""
    
    #### METHOD CHOICE ####
    method_choice = 'auto'
    
    #### MODEL CHOICE ####
    model_choice = CompositePolytropeConfig(
        indices=3.0,
        target_pressures=-np.inf,
    )
    # model_choice = CompositePolytropeConfig(
    #     indices=(2.0, 1.0, 3.0, 1.5, 2.0, 4.0),
    #     target_pressures=(-1.0, -2.0, -3.0, -5.0, -7.0, -np.inf),
    #     density_jumps=(0.3, 0.2, 2.0, 0.5, 0.2),
    # )
    # model_choice = 'Jupiter.txt'

    #### ROTATION PARAMETERS ####      
    rotation_profile = solid
    rotation_target = 0.9
    central_diff_rate = 1.0
    rotation_scale = 1.0
    
    #### SOLVER PARAMETERS ####
    max_degree = angular_resolution = 101
    full_rate = 3
    mapping_precision = 1e-10
    lagrange_order = 3
    spline_order = 5
    
    #### OUTPUT OPTIONS ####
    output_options = OutputOptions(
        diagnostics=DiagnosticOptions(
            virial_test=False,
            gravitational_moments=False,
        ),
        plot=PlotOptions(
            show_harmonics=False,
            show_model=True,
            resolution=501,
            surfaces=True,
        ),
        flux=RadiativeFluxOptions(
            enabled=False,
            plot_lines=True,
            origin=0.05,
            n_lines=15,
            show_effective_temperature=True,
            resolution=(200, 100),
        ),
        model=ModelOutputOptions(
            save=False,
            dimensional=False,
        ),
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
        output_options, 
        external_domain_res, rescale_ab
    )
    
if __name__ == '__main__' :
    
    # Setting the global parameters
    method_choice, model_choice, rotation_profile, rotation_target,     \
    central_diff_rate, rotation_scale, max_degree, angular_resolution,  \
    full_rate, mapping_precision, spline_order, lagrange_order,         \
    output_options, external_domain_res, rescale_ab = set_params()
    
    # Choosing the method to call
    if method_choice == "auto":
        if isinstance(model_choice, CompositePolytropeConfig):
            n_domains = model_choice.n_regions
        else:
            r = np.genfromtxt(
                Path("Models") / model_choice,
                skip_header=2,
                usecols=0,
            )
            n_domains = find_domains(r).n_domains

        method_choice = "spheroidal" if n_domains > 1 else "radial"

    solvers = {
        "radial": radial_method,
        "spheroidal": spheroidal_method,
    }

    try:
        solver = solvers[method_choice]
    except KeyError:
        raise ValueError(
            f"Unknown method {method_choice!r}; expected "
            "'auto', 'radial' or 'spheroidal'."
        ) from None

    solver(
        model_choice,
        rotation_profile,
        rotation_target,
        central_diff_rate,
        rotation_scale,
        max_degree,
        angular_resolution,
        full_rate,
        mapping_precision,
        spline_order,
        lagrange_order,
        output_options,
        external_domain_res,
        rescale_ab,
    )
