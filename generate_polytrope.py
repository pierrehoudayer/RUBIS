#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:06:45 2023

@author: phoudayer
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import solve_ivp
from dotdict import DotDict

def polytrope(N, P0=0.0, R=1.0, M=1.0, res=1001) :
    """
    Generate a polytrope of given radius, mass and surface pressure.
    
    Parameters
    ----------
    N : int
        Polytrope index
    P0 : float
        Surface pressure
    R : float, optional
        Polytrope radius. The default value is 1.0
    M : float, optional
        Polytrope mass. The default value is 1.0
    res : int, optional
        Number of points. The default value is 1001
        
    Returns
    -------
    model : Dotdict instance
        Dictionary containing the model variables : {
            r : array_like, shape (res, )
                Radial coordinate
            P : array_like, shape (res, )
                Pressure
            rho : array_like, shape (res, )
                Density
            g : array_like, shape (res, )
                Gravity
        }
    """
    
    if N in {1.0} :             # The analytical solution is known
        
        # Solution
        @np.errstate(all='ignore')
        def df(x) : return (np.cos(x) - np.sinc(x / np.pi)) / x
        f  = lambda x : np.sinc(x / np.pi)
        x0, g0 = np.pi, 2 / np.pi
        
    else :
        
        # Solver arguments
        Xo = 20.0
        solver_method = 'DOP853'
        tol = 1e-13
        
        # Differential equation dy_dx = f(x, y)
        def differential_equation(x, y) : 
            f = np.empty_like(y)
            with np.errstate(all='ignore') :
                f[0] = np.where(
                    x > 1e-8, (-2) * y[0]/x - y[1]**N, (-1/3) * y[1]**N
                    )
            f[1] = y[0]
            return f
        
        # Reaching surface event
        surface = lambda x, y : y[1] - P0**(1/(N+1))
        surface.terminal = True 
        
        # Actual solving
        reach_surface = False
        while not reach_surface : 
            sol = solve_ivp(
                differential_equation, (0, Xo), (0.0, 1.0), method=solver_method, 
                events=surface, dense_output=True, rtol=tol, atol=tol
                )
            reach_surface = bool(sol.status)
            Xo *= 2.0
            
        # Solution
        df = lambda x : sol.sol(x)[0] 
        f  = lambda x : np.abs(sol.sol(x)[1])
        x0, g0 = sol.t[-1], -(N+1) * sol.y[0, -1]
    
        
    # Scales determination
    G = 6.67384e-8                      # Gravitational constant
    L_scale = R / x0
    G_scale = (G*M/R**2) / g0
    hc   = L_scale * G_scale
    rhoc = (N+1)/(4*np.pi*G) * (G_scale / L_scale)
    pc   = rhoc * hc
    
    # Rescaling
    x  = x0 * np.sin(np.linspace(0, np.pi/2, res))
    h, dh  = f(x), df(x)
    
    # Defining model
    model = DotDict()
    model.r   = L_scale * x
    model.g   = G_scale * (N+1) * (-dh)
    model.rho = rhoc    * h**N
    model.p   = pc      * h**(N+1) 
    
    return model

def composite_polytrope(model_parameters) :
    """
    Generate a composite polytrope of given radius and mass.
    The latter is composed of N polytropes, the i-th one having an index n_i = indices[i].
    The i-th interface is located at a pressure value verifying: 
        log10(P / P_center) = target_pressures[i-1]    <-- Python indexing convention
    and the N-th value determines the surface pressure:
        log10(P_surf / P_center) = target_pressures[N-1].
    If one wants the surface pressure to be 0.0, then target_pressures[N-1] must be -np.inf.
    Each interface (surface excluded) possesses a "density jump" defined as:
        rho_i(r+) = rho_{i-1}(r-) * density_jumps[i]
    and which can be greater than 1 (density returnal).
    
    Note: A standard polytrope (of index n) can be retrieved using:
        model = composite_polytrope(indices = n, target_pressures = -np.inf)
    
    Parameters
    ----------
    model_parameters : DotDict instance containing: {
        indices : array_like, shape(N, )
            Each region polytropic index
        target_pressures : array_like, shape(N, )
            Normalised interface pressure values (surface included).
        density_jumps : array_like, shape(N-1, )
            Density ratios above and below each interface (surface excluded).
            The default value is np.ones((number_of_regions-1,))
        R : float, optional
            Composite polytrope radius. Set to 1.0 if None
        M : float, optional
            Composite polytrope mass. Set to 1.0 if None
        res : int, optional
            Number of points. Set to 1001 if None
    }
        
    Returns
    -------
    model : Dotdict instance
        Dictionary containing the model variables : {
            r : array_like, shape (res, )
                Radial coordinate
            P : array_like, shape (res, )
                Pressure
            rho : array_like, shape (res, )
                Density
            g : array_like, shape (res, )
                Gravity
        }
    """
    # Dictionary reading
    indices           = np.atleast_1d(model_parameters.indices)
    number_of_regions = len(indices)
    target_pressures  = np.hstack((0.0, model_parameters.target_pressures))
    density_jumps     = model_parameters.density_jumps or np.ones((number_of_regions-1,))
    R                 = model_parameters.R or 1.0
    M                 = model_parameters.M or 1.0
    res               = model_parameters.res or 1001
    
    # Solver arguments
    dxi_est = 20.0
    solver_method = 'DOP853'
    tol = 1e-13
    
    # Differential equation dy_dx = f(x, y)
    def differential_equation(x, y, Ni, Ci) : 
        f = np.empty_like(y)
        f[0] = y[1]
        with np.errstate(all='ignore') :
            f[1] = np.where(
                x > 1e-8, (-2) * y[1]/x - Ci * y[0]**Ni, (-1/3) * Ci * y[0]**Ni
            )
        return f
    
    # interface function
    def interface(x, y, Ni, Ei) :
        return y[0] - 10 ** (Ei / (Ni+1))
    
    # Find the solution for each region
    solutions = []
    constants = []
    for i in range(number_of_regions) :
        # Region initialisation
        if i > 0 : 
            ui_1, vi_1 = solutions[i-1].y[:, -1]
            Ci_1 = constants[i-1]
            xi_1 = solutions[i-1].t[-1]
            Ei_1, Ei = np.diff(target_pressures)[i-1:i+1]
            Ni_1, Ni = indices[i-1:i+1]
            pi = density_jumps[i-1]
            vi = (Ni_1+1)/(Ni+1) * pi**1 * 10**-Ei_1 * ui_1**(1*Ni_1) * vi_1
            Ci = (Ni_1+1)/(Ni+1) * pi**2 * 10**-Ei_1 * ui_1**(2*Ni_1) * Ci_1
        else : 
            xi_1 = 0.0
            Ei = np.diff(target_pressures)[i]
            Ni = indices[i]
            vi = 0.0
            Ci = 1.0
            
        # Define the current differential equation
        current_differential_equation = lambda x, y : differential_equation(x, y, Ni, Ci)
        
        # Define the reaching interface event
        reaching_interface = lambda x, y : interface(x, y, Ni, Ei)
        reaching_interface.terminal = True
        
        # Actual solving
        event_reached = False
        while not event_reached : 
            sol = solve_ivp(
                current_differential_equation, (xi_1, xi_1 + dxi_est), (1.0, vi), 
                method=solver_method, events=reaching_interface, dense_output=True, 
                rtol=tol, atol=tol
            )
            event_reached = bool(sol.status)
            if not event_reached : dxi_est *= 2.0
            
        # Update the solution
        solutions.append(sol)
        constants.append(Ci)
        
    # Collect values on the interfaces
    x_out, v_out = sol.t[-1], sol.y[1, -1]
    x_int = np.array([sol.t[0] for sol in solutions] + [x_out])
    
    # Define the final grid
    x_est = x_out * np.sin(np.linspace(0, np.pi/2, res - 2*number_of_regions + 2))
    x_per_domain = [
        np.hstack((
            x_int[i], x_est[(x_est > x_int[i])&(x_est < x_int[i+1])], x_int[i+1]
        )) for i in range(number_of_regions)
    ]
    
    # Solution rescaling
    G = 6.67384e-8                      # Gravitational constant
    r_scale = R / x_out
    model = DotDict()
    model.rho = []
    model.p   = []   
    model.g   = []   
    for i in range(number_of_regions-1, -1, -1) :
        # Scaling computation
        Ni = indices[i]
        if i < number_of_regions-1 : 
            Ei = np.diff(target_pressures)[i]
            pi = density_jumps[i]
            rho_scales.append(10 ** (-Ei*Ni/(Ni+1)) / pi * rho_scales[-1])
            p_scales.append(  10 ** (-Ei)                * p_scales[-1])
            g_scales.append(  10 ** (-Ei   /(Ni+1)) * pi * g_scales[-1])
            
        else :
            rho_scales = [- (x_out / (4*np.pi*v_out))     * (  M   /R**3)]
            p_scales   = [+ (4*np.pi*(Ni+1)*v_out**2)**-1 * (G*M**2/R**4)]
            g_scales   = [- (v_out*(Ni+1))**-1            * (G*M   /R**2)]
            
        # Current solution
        yi = solutions[i].sol(x_per_domain[i])
        
        # Scaled variables
        model.rho = np.hstack((rho_scales[-1] * yi[0]**(Ni+0), model.rho))
        model.p   = np.hstack((  p_scales[-1] * yi[0]**(Ni+1), model.p))
        model.g   = np.hstack((- g_scales[-1] * yi[1] *(Ni+1), model.g))
    model.r   = r_scale * np.hstack(x_per_domain) 
    
    return model


if __name__ == '__main__':
    # Model creation
    model_parameters = DotDict(
        indices = (6.0, 1.0, 3.0, 1.5, 2.0, 4.0), 
        target_pressures = (-1.0, -2.0, -3.0, -5.0, -7.0, -np.inf), 
        density_jumps = (0.3, 0.2, 1.2, 0.8, 0.2)
    )
    model = composite_polytrope(model_parameters)
    
    # Find the interfaces
    _, idx = np.unique(model.r, return_index=True)
    idxref = np.arange(len(model.r))
    interfaces = model.r[list(set(idxref) - set(idx))]
    
    # Plot
    size = 16
    ymax = np.max([np.max(model.rho), np.max(model.p), np.max(model.g)]) * 10
    rc('text', usetex=True)
    rc('xtick', labelsize=size)
    rc('ytick', labelsize=size)
    plt.plot(model.r, model.p,   label=r"$P$")
    plt.plot(model.r, model.rho, label=r"$\rho$")
    plt.plot(model.r, model.g,   label=r"$g$")
    plt.ylim((1e-15, ymax))
    plt.vlines(interfaces, ymin=1e-15, ymax=ymax, colors='grey', alpha=0.3, linewidth=1.0)
    plt.legend(fontsize=size)
    plt.yscale('log')
    plt.show()
    
    
    
    