# def find_centrifugal_potential(r, cth, omega, dim=False) :
#     """
#     Determination of the centrifugal potential and its 
#     derivative in the case of a cylindric rotation profile 
#     (caracterised by ALPHA). The option dim = True allows a
#     computation taking into account the future renormalisation
#     (in this case r_eq != 1 but r_eq = R_eq / R).

#     Parameters
#     ----------
#     r : float or array_like, shape (Nr, )
#         Radial value(s).
#     cth : float or array_like, shape (Nr, )
#         Value(s) of cos(theta).
#     omega : float
#         Rotation rate.
#     dim : boolean, optional
#         Set to true for the omega computation. 
#         The default is False.

#     Returns
#     -------
#     phi_c : float or array_like, shape (Nr, )
#         Centrifugal potential.
#     dphi_c : float or array_like, shape (Nr, )
#         Centrifugal potential derivative with respect to r.

#     """
#     phi_c, dphi_c = eval_phi_c(r, cth, omega)
#     if dim :
#         return phi_c / r**3, (r*dphi_c - 3*phi_c) / r**4
#     else :
#         return phi_c, dphi_c


# def estimate_omega(phi_g, phi_g_l_surf, target, omega) :
#     """
#     Estimates the adequate rotation rate so that it reaches ROT
#     after normalisation. Considerably speed-up (and stabilises)
#     the overall convergence.

#     Parameters
#     ----------
#     phi_g : function(r_eval)
#         Gravitational potential along the equatorial cut.
#     phi_g_l_surf : array_like, shape (L, )
#         Gravitation potential harmonics on the surface.
#     target : float
#         Value to reach for the effective potential.
#     omega_n : float
#         Current rotation rate.

#     Returns
#     -------
#     omega_n_new : float
#         New rotation rate.

#     """    
#     # Searching for a new omega
#     l    = np.arange(L)
#     dr    = 1.0
#     r_est = 1.0
#     while abs(dr) > DELTA : 
#         # Star's exterior
#         if r_est >= r[-1] :
#             phi_g_l_ext  = phi_g_l_surf * (1.0/r_est)**(l+1)
#             dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_est
#             phi_g_est  = pl_eval_2D( phi_g_l_ext, 0.0)
#             dphi_g_est = pl_eval_2D(dphi_g_l_ext, 0.0)
            
#         # Star's interior
#         else :
#             phi_g_est  = phi_g(r_est, nu=0)
#             dphi_g_est = phi_g(r_est, nu=1)
        
#         # Centrifugal potential 
#         phi_c_est, dphi_c_est = find_centrifugal_potential(
#             r_est, 0.0, omega_n, dim=True
#         )
        
#         # Total potential
#         phi_t_est  =  phi_g_est +  phi_c_est
#         dphi_t_est = dphi_g_est + dphi_c_est
        
#         # Update r_est
#         dr = (target - phi_t_est) / dphi_t_est
#         r_est += dr
        
#     # Updating omega
#     omega_n_new = omega_n * r_est**(-1.5)
#     return omega_n_new


# def find_new_mapping(map_n, omega_n, phi_g_l, dphi_g_l, phi_eff) :
#     """
#     Find the new mapping by comparing the effective potential
#     and the total potential (calculated from phi_g_l and omega_n).

#     Parameters
#     ----------
#     map_n : array_like, shape (N, M)
#         Current mapping.
#     omega_n : float
#         Current rotation rate.
#     phi_g_l : array_like, shape (N, L)
#         Gravitation potential harmonics.
#     dphi_g_l : array_like, shape (N, L)
#         Gravitation potential derivative harmonics.
#     phi_eff : array_like, shape (N, )
#         Effective potential on each equipotential.

#     Returns
#     -------
#     map_n_new : array_like, shape (N, M)
#         Updated mapping.
#     omega_n_new : float
#         Updated rotation rate.

#     """
    # # 2D gravitational potential
    # phi2D_g =  pl_eval_2D( phi_g_l, cth)
    # dphi2D_g = pl_eval_2D(dphi_g_l, cth)
        
    # # Gravitational potential interpolation
    # l, up = np.arange(L), np.arange((M+1)//2)
    # phi_g_func = [CubicHermiteSpline(
    #     x=r, y=phi2D_g[:, k], dydx=dphi2D_g[:, k]
    # ) for k in up]
    
    # # Find a new value for ROT
    # target = phi_eff[N-1] - pl_eval_2D(phi_g_l[0], 0.0) + phi_eff[0]
    # omega_n_new = estimate_omega(phi_g_func[-1], phi_g_l[-1], target, omega_n)
    
    # # Find the mapping
    # map_est = np.copy(map_n[:, up])
    # idx = np.indices(map_est.shape)
    # dr = np.ones_like(map_n[:, up])
    # dr[0] = 0.0        
    
    # while np.any(np.abs(dr) > DELTA) :
        
    #     # Star's interior
    #     C_int = (np.abs(dr) > DELTA) & (map_est <  1.0)
    #     r_int = map_est[C_int]
    #     k_int = idx[1,  C_int]
        
    #     if 0 not in k_int.shape :
            
    #         # Gravitational potential
    #         inv_sort = np.argsort(app_list(np.arange(len(k_int)), k_int))
    #         phi_g_int  = np.array(
    #             (app_list(r_int, k_int, phi_g_func        ),
    #              app_list(r_int, k_int, phi_g_func, (1,)*M))
    #         )[:, inv_sort]
            
    #         # Centrifugal potential
    #         phi_c_int = np.array(
    #             find_centrifugal_potential(r_int, cth[k_int], omega_n_new)
    #         )
            
    #         # Total potential
    #         phi_t_int =  phi_g_int +  phi_c_int
            
    #         # Update map_est
    #         dr[C_int] = (phi_eff[idx[0, C_int]] - phi_t_int[0]) / phi_t_int[1]
    #         if np.any(np.abs(dr[C_int]) > 2.0) :
    #             print("INT")
    #             plt.scatter(r_int, np.abs(cth[k_int]), 
    #                 c=1.0/phi_t_int[1], cmap=cm.RdBu_r, edgecolors='k', 
    #                 linewidths=0.05, norm=mc.CenteredNorm()
    #             )
    #             plt.colorbar()
    #             plt.ylim(0.0, 1.0)
    #             plt.show()
    #         # map_est[C_int] = np.abs(map_est[C_int] + dr[C_int])
    #         map_est[C_int] = map_est[C_int] + dr[C_int]
            
    #     # Star's exterior
    #     C_ext = (np.abs(dr) > DELTA) & (map_est >=  1.0)
    #     r_ext = map_est[C_ext]
    #     k_ext = idx[1,  C_ext]
        
    #     if 0 not in k_ext.shape :
            
    #         # Gravitational potential
    #         phi_g_l_ext  = phi_g_l[-1] * (r_ext[:, None])**-(l+1)
    #         dphi_g_l_ext = -(l+1) * phi_g_l_ext / r_ext[:, None]
    #         inv_sort = np.argsort(app_list(np.arange(len(k_ext)), k_ext))
    #         phi_g_ext  = np.vstack(
    #             [app_list(harms, k_ext, pl_eval_2D, cth)
    #              for harms in (phi_g_l_ext, dphi_g_l_ext)]
    #         )[:, inv_sort]
            
    #         # Centrifugal potential
    #         phi_c_ext = np.array(
    #             find_centrifugal_potential(r_ext, cth[k_ext], omega_n_new)
    #         )
            
    #         # Total potential
    #         phi_t_ext =  phi_g_ext +  phi_c_ext
            
    #         # Update map_est
    #         dr[C_ext] = (phi_eff[idx[0, C_ext]] - phi_t_ext[0]) / phi_t_ext[1]
    #         if np.any(np.abs(dr[C_ext]) > 2.0) :
    #             print("EXT")
    #             plt.scatter(r_ext, np.abs(cth[k_ext]), 
    #                 c=1.0/phi_t_ext[1], cmap=cm.RdBu_r, edgecolors='k', 
    #                 linewidths=0.05, norm=mc.CenteredNorm()
    #             )
    #             plt.colorbar()
    #             plt.ylim(0.0, 1.0)
    #             plt.show()
                
    #         # map_est[C_ext] = np.abs(map_est[C_ext] + dr[C_ext])
    #         map_est[C_ext] = map_est[C_ext] + dr[C_ext]
    
    # # New mapping
    # map_n_new = np.hstack((map_est, np.flip(map_est, axis=1)[:, 1:]))
    # return map_n_new


# def app_list(val, idx, func=lambda x: x, args=None) :
#     """
#     Function only designed for convenience in the vectorial mapping finding.

#     Parameters
#     ----------
#     val : list
#         Values on which func is applied
#     idx : list
#         val ordering.
#     func : function or list of function, optional
#         functions to be applied on val. The default is lambda x: x (identity).
#     args : list, optional
#         list of function arguments. The default is None.

#     Returns
#     -------
#     array_like
#         The function applied to val with corresponding args.

#     """
#     unq_idx = np.unique(idx)
#     if (args is None) & (callable(func)) :
#         Func = lambda l: func(val[idx == l])
#     elif (args is None) & (not callable(func)) :
#         Func = lambda l: func[l](val[idx == l])
#     elif (args is not None) & (callable(func)) :
#         Func = lambda l: func(val[idx == l], args[l])
#     else :
#         Func = lambda l: func[l](val[idx == l], args[l])
#     return np.hstack(list(map(Func, unq_idx)))

# def mapdiff(map_sph, map_rad, L,
#             size=16, n_lev=201, res=501, cmap=cm.seismic) : 
    
#     N, M = map_rad.shape
#     cth_res = np.linspace(-1, 1, res)
#     sth_res = np.sqrt(1-cth_res**2)
#     map_l_sph = pl_project_2D(map_sph, L)
#     map_l_rad = pl_project_2D(map_rad, L)
#     map_res_sph = pl_eval_2D(map_l_sph, cth_res)
#     map_res_rad = pl_eval_2D(map_l_rad, cth_res)
#     diff = map_res_rad - map_res_sph
    
#     plt.close('all')
#     plt.contourf(
#         map_res_rad*sth_res, map_res_rad*cth_res, diff, 
#         cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
#     )
#     plt.contourf(
#         -map_res_rad*sth_res, map_res_rad*cth_res, diff, 
#         cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
#     )
#     plt.plot( map_res_rad[-1]*sth_res, map_res_rad[-1]*cth_res, 'k--',lw=0.5)
#     plt.plot(-map_res_rad[-1]*sth_res, map_res_rad[-1]*cth_res, 'k--',lw=0.5)
#     plt.xlabel(r"$s/R_{\mathrm{eq}}$",fontsize=size)
#     plt.ylabel(r"$z/R_{\mathrm{eq}}$",fontsize=size)
#     cbar = plt.colorbar()
#     cbar.set_label(r"$\delta r$", fontsize=size)
#     plt.gca().set_aspect("equal")
#     plt.show()

# def phidiff(phi_l_sph, phi_l_rad, map_sph, r,
#             size=16, n_lev=201, res=501, cmap=cm.seismic) : 
    
#     N, L = phi_l_rad.shape
#     dr_sph = find_metric_terms(map_sph)
#     dr_sph = find_external_mapping(dr_sph)
    
#     cth_res = np.linspace(-1, 1, res)
#     sth_res = np.sqrt(1-cth_res**2)
#     map_l_sph = pl_project_2D(dr_sph._, L)
#     map_res = pl_eval_2D(map_l_sph, cth_res)
    
#     phi2D_sph = pl_eval_2D(phi_l_sph, cth_res)
#     phi2D_rad = pl_eval_2D(phi_l_rad, cth_res)
#     l = np.arange(L)
#     phi2D_int = np.array(
#         [np.hstack(
#             (interpolate_func(r, phik)(rk[rk < 1]), 
#              pl_eval_2D(phi_l_rad[-1] * (rk[rk >= 1, None])**-(l+1), ck))
#             )
#          for rk, ck, phik in zip(map_res.T, cth_res, phi2D_rad.T)]
#         ).T
#     diff = phi2D_int - phi2D_sph
    
#     plt.close('all')
#     plt.contourf(
#         map_res*sth_res, map_res*cth_res, diff, 
#         cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
#     )
#     plt.contourf(
#         -map_res*sth_res, map_res*cth_res, diff, 
#         cmap=cmap, levels=n_lev, norm=mcl.CenteredNorm()
#     )
#     plt.plot( map_res[N-1]*sth_res, map_res[N-1]*cth_res, 'k-',lw=0.5)
#     plt.plot(-map_res[N-1]*sth_res, map_res[N-1]*cth_res, 'k-',lw=0.5)
#     plt.plot( map_res[-1]*sth_res, map_res[-1]*cth_res, 'k--',lw=0.5)
#     plt.plot(-map_res[-1]*sth_res, map_res[-1]*cth_res, 'k--',lw=0.5)
#     plt.xlabel(r"$s/R_{\mathrm{eq}}$",fontsize=size)
#     plt.ylabel(r"$z/R_{\mathrm{eq}}$",fontsize=size)
#     cbar = plt.colorbar()
#     cbar.set_label(r"$\delta \phi_g$", fontsize=size)
#     plt.gca().set_aspect("equal")
#     plt.show()

# def phieff_lines(phi_g_l, dphi_g_l, r, zeta, omega, eval_phi_c,
#                  size=16, n_lev=51, res=501, cmap=cm.seismic) : 
    
#     N, _ = r.shape
#     _, L = phi_g_l.shape
#     z = zeta[N:].reshape((-1, 1))
        
#     surf = r[-1]
#     max_degree = 3
#     r_ext = np.vstack((r, z - (1-surf)*(2-z)**max_degree))
    
#     cth_res = np.linspace(-1, 1, res)
#     sth_res = np.sqrt(1-cth_res**2)
#     map_l = pl_project_2D(r_ext, L)
#     map_res = pl_eval_2D(map_l, cth_res)
    
#     phi2D_g = pl_eval_2D(phi_g_l, cth_res)
#     phi2D_c = np.array(
#         [eval_phi_c(rk, ck, omega)[0] for rk, ck in zip(map_res.T, cth_res)]
#     ).T
#     phi2D_eff = phi2D_g + phi2D_c
    
#     _, dom_unq = np.unique(zeta, return_index=True)
#     phi_g_eq  = interpolate_func(
#         pl_eval_2D(map_l  , 0.0)[dom_unq], 
#         pl_eval_2D(phi_g_l, 0.0)[dom_unq]
#     )
#     dphi_g_eq = interpolate_func(
#         pl_eval_2D(map_l   , 0.0)[dom_unq], 
#         pl_eval_2D(dphi_g_l, 0.0)[dom_unq]
#     )
#     f  = lambda r :  phi_g_eq(r)  + eval_phi_c(r, 0.0, omega)[0]
#     df = lambda r : dphi_g_eq(r)  + eval_phi_c(r, 0.0, omega)[1]
#     from scipy.optimize import root_scalar
    
#     r_crit = root_scalar(df, bracket=[1.0, 2.0]).root
#     phi_eff_crit = f(r_crit)
#     phi2D_eff -= phi_eff_crit
    
#     phimax, phimin = phi2D_eff.max(), phi2D_eff.min()
#     levels = (
#         list(np.flip(phimin * np.linspace(0.0, 1.0, (n_lev+1)//2) ** 2))[:-1] +
#         list(phimax * np.linspace(0.0, 1.0, (n_lev+1)//2) ** 2)
#     )
#     lw = 0.5 * np.ones(n_lev)
#     lw[(n_lev - 1)//2] = 2.0
#     if isinstance(cmap, list) :
#         cmap = get_continuous_cmap(cmap)
    
#     rc('text', usetex=True)
#     rc('xtick', labelsize=size)
#     rc('ytick', labelsize=size)
#     plt.close('all')
#     plt.contour(
#         map_res*sth_res, map_res*cth_res, phi2D_eff, 
#         cmap=cmap, levels=levels, norm=mcl.CenteredNorm(), linewidths=lw
#     )
#     plt.contour(
#         -map_res*sth_res, map_res*cth_res, phi2D_eff, 
#         cmap=cmap, levels=levels, norm=mcl.CenteredNorm(), linewidths=lw
#     )
#     plt.xlabel(r"$s/R_{\mathrm{eq}}$",fontsize=size)
#     plt.ylabel(r"$z/R_{\mathrm{eq}}$",fontsize=size)
#     plt.gca().set_aspect("equal")
#     plt.axis("off")
#     plt.show()


    # ### TESTS (DELME)
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcl
    # import matplotlib.cm as cm
    # from scipy.interpolate import CubicHermiteSpline
    # from scipy.special import roots_legendre
    # from legendre  import pl_eval_2D, pl_project_2D
    # from numerical import interpolate_func
    # from helpers import find_domains, plot_f_map
    
    # N = len(r)
    # L, M = max_degree, angular_resolution
    # eq = (M-1)//2
    # t, weights = roots_legendre(M)
    # dom = find_domains(zeta)
    
    # Del = lambda x, f, der: interpolate_func(x, f, der=der, k=5)(x)
    
    # Sph = lambda func, *args, **kwargs: np.hstack(
    #     [func(*[a[D] for a in args], **kwargs) for D in dom.ranges[:-1]]
    # )
    
    # # metric terms
    # dr = DotDict()
    # dr._ = map_n
    # map_l = pl_project_2D(dr._, L)
    # _, dr.t = pl_eval_2D(map_l, t, der=1)
    # if dom.Nd == 1 : 
    #     dr.z = np.array([Del(zeta, rk, der=1) for rk in map_n.T]).T 
    #     dP_dz = Del(zeta, P, der=1)
    # else :
    #     dr.z = np.array([Sph(Del, zeta, rk, der=1) for rk in map_n.T]).T
    #     dP_dz = Sph(Del, zeta, P, der=1)
    
    # dP_dr = + (1.0  / dr.z) * dP_dz[:, None]
    # dP_dt = - (dr.t / dr.z) * dP_dz[:, None]
    
    # phi_g = pl_eval_2D(phi_g_l, t)
    # _, dphi_g_dt = pl_eval_2D(phi_g_l, t, der=1)
    # dphi_g_dr = pl_eval_2D(dphi_g_l, t)
    # if dom.Nd == 1 : 
    #     dphi_g_dr = np.array(
    #         [CubicHermiteSpline(r, pk, dpk)(rk, nu=1) for rk, pk, dpk in zip(map_n.T, phi_g.T, dphi_g_dr.T)]
    #     ).T
    #     dphi_g_dt = np.array(
    #         [interpolate_func(r, dpk, k=5)(rk) for rk, dpk in zip(map_n.T, dphi_g_dt.T)]
    #     ).T
    # else :
    #     dphi_g_dr[:N] /= dr.z
    
    # rota2D = np.array([eval_w(rk, tk, rotation_target) for rk, tk in zip(map_n.T, t)]).T
    
    # EQ_r = dP_dr + rho[:, None] * (dphi_g_dr[:N] - rota2D[:N]**2 * map_n[:N] * (1-t**2))
    # EQ_t = dP_dt + rho[:, None] * (dphi_g_dt[:N] + rota2D[:N]**2 * map_n[:N]**2 * t    )
    
    # # N_r = np.abs(dP_dr) + np.abs(rho[:, None] * dphi_g_dr) + np.abs(rho[:, None] * rota2D**2 * map_n * (1-t**2))
    # # N_t = np.abs(dP_dt) + np.abs(rho[:, None] * dphi_g_dt) + np.abs(rho[:, None] * rota2D**2 * map_n**2 * t)
    
    # # EQ_r = np.divide(EQ_r, N_r, out=np.zeros_like(EQ_r), where=(N_r!=0.0)&(np.abs(N_r)!=1.0))
    # # EQ_t = np.divide(EQ_t, N_t, out=np.zeros_like(EQ_t), where=(N_t!=0.0)&(np.abs(N_t)!=1.0))
    
    
    # plot_f_map(
    #     map_n, np.log10(np.abs(EQ_r)), phi_eff, L, 
    #     angular_res=output_params.plot_resolution,
    #     cmap='viridis',
    #     show_surfaces=False,
    #     label=r"$\delta \mathrm{EQ}_r$",
    #     disc=dom.end[:-1]
    # )
    # plot_f_map(
    #     map_n, np.log10(np.abs(EQ_t)+1e-15), phi_eff, L, 
    #     angular_res=output_params.plot_resolution,
    #     cmap='cividis',
    #     show_surfaces=False,
    #     label=r"$\delta \mathrm{EQ}_t$",
    #     disc=dom.end[:-1]
    # )