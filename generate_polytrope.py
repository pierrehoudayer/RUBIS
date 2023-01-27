#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:06:45 2023

@author: phoudayer
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dotdict import DotDict

def polytrope(N, P0, R=1.0, M=1.0, res=1001) :
    """
    Generate a polytrope of given radius, mass and surface
    pressure.
    
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
        
    # Scales determination
    G = 6.67384e-8                      # Gravitational constant
    x0, g0 = sol.t[-1], -(N+1) * sol.y[0, -1]
    L_scale = R / x0
    G_scale = (G*M/R**2) / g0
    hc   = L_scale * G_scale
    rhoc = (N+1)/(4*np.pi*G) * (G_scale / L_scale)
    pc   = rhoc * hc
    
    # Rescaling
    x  = sol.t[-1] * np.sin(            # Denser grid on surface
        np.linspace(0, np.pi/2, res)
    )
    h  = np.abs(sol.sol(x)[1])
    dh = sol.sol(x)[0]
    
    # Defining model
    model = DotDict()
    model.r   = L_scale * x
    model.g   = G_scale * (N+1) * (-dh)
    model.rho = rhoc    * h**N
    model.p   = pc      * h**(N+1) 
    
    return model


if __name__ == '__main__':
    beg = time.perf_counter()
    model = polytrope(1, 0.0)
    end = time.perf_counter()
    print(f"Chrono : {np.round(end-beg, 8)}s")
    plt.plot(model.r, model.p, marker='.')
    
    
    
    