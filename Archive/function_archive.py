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




# def find_metric_terms(map_n, z0=0.0, z1=1.0) : 
#     """
#     Finds the metric terms, i.e the derivatives of r(z, t) 
#     with respect to z or t (with z := zeta and t := cos(theta)),
#     for z0 <= z <= z1.

#     Parameters
#     ----------
#     map_n : array_like, shape (N, M)
#         Isopotential mapping.

#     Returns
#     -------
#     dr : DotDict instance
#         The mapping derivatives : {
#             _   = r(z, t),
#             t   = r_t(z, t),
#             tt  = r_tt(z, t),
#             z   = r_z(z, t),
#             zt  = r_zt(z, t),
#             ztt = r_ztt(z, t),
#             zz  = r_zz(z, t),
#             S   = \Delta_S r(z, t)
#             }          
#     """
#     N0 = np.argwhere(zeta >= z0)[0, 0]
#     N1 = np.argwhere(zeta >= z1)[0, 0] + 1
#     dr = DotDict()
#     dr._ = map_n[N0:N1]
#     map_l = pl_project_2D(dr._, L)
#     _, dr.t, dr.tt = pl_eval_2D(map_l, cth, der=2)
#     dr.z = np.array(
#         [interpolate_func(zeta[N0:N1], rk, der=1, k=KSPL)(zeta[N0:N1]) for rk in dr._.T]
#     ).T 
#     map_l_z = pl_project_2D(dr.z, L)
#     _, dr.zt, dr.ztt = pl_eval_2D(map_l_z, cth, der=2)
#     dr.zz = np.array(
#         [interpolate_func(zeta[N0:N1], rk, der=2, k=KSPL)(zeta[N0:N1]) for rk in dr._.T]
#     ).T 
#     dr.S = (1-cth**2) * dr.tt - 2*cth * dr.t
#     return dr


# def add_advanced_metric_terms(dr) : 
#     """
#     Finds the more advanced metric terms, useful in very specific cases.

#     Parameters
#     ----------
#     dr : DotDict instance
#         The object containing the mapping derivatives.

#     Returns
#     -------
#     dr : DotDict instance
#         Same object but enhanced with the following terms : {
#             c2 = cos(b^z, b_z) ** 2 : 
#                 squared cosinus between the covariant and 
#                 contravariant zeta vectors in the natural basis.
#             cs = cos(b^z, b_z) * sin(b^z, b_z) :
#                 cosinus by sinus of the same angle.
#                 /!\ The orientation of this angle has been chosen
#                     to be the same as theta (i.e. inverse trigonometric)
#             gzz : zeta/zeta covariant metric term.
#             gzt : zeta/theta covariant metric term.
#             gtt : theta/theta covariant metric term.
#             gg = gzt / gzz : 
#                 covariant ratio.
#                 /!\ The latter has been multiplied by (1 - cth**2) ** 0.5
#             divz = div(b^z) : 
#                 divergence of the zeta covariant vector
#             divt = div(b^t) : 
#                 divergence of the theta covariant vector
#             divrelz = div(b^z) / gzz : 
#                 relative divergence of the zeta covariant vector
#             divrelt = div(b^t) / gtt : 
#                 relative divergence of the theta covariant vector
#         }          
#     """
#     # Trigonometric terms
#     with np.errstate(all='ignore'):
#         dr.c2 = np.where(
#             dr._ == 0.0, 1.0, dr._ ** 2 / (dr._ ** 2 + (1-cth**2) * dr.t ** 2)
#         )
#         dr.cs = np.where(
#             dr._ == 0.0, 0.0, 
#             (1-cth**2) ** 0.5 * dr._ * dr.t / (dr._ ** 2 + (1-cth**2) * dr.t ** 2)
#         )
        
#     # Covariant metric terms
#     dr.gzz = 1.0 / (dr.z ** 2 * dr.c2)
#     with np.errstate(all='ignore'):
#         dr.gzt = np.where(
#             dr._ == 0.0, np.nan, (1-cth**2) ** 0.5 * dr.t / (dr.z * dr._ ** 2)
#         )
#         dr.gtt = np.where(
#             dr._ == 0.0, np.nan, dr._ ** (-2)
#         )
#         dr.gg  = np.where(
#             dr._ == 0.0, np.nan, 
#             - (1-cth**2) * dr.z * dr.t / (dr._ ** 2 + (1-cth**2) * dr.t ** 2) 
#         )
    
#     # Divergence
#     with np.errstate(all='ignore'):
#         dr.divz = (
#             np.where(
#                 dr._ == 0.0, np.nan, 
#                 (2 * dr._ + 2 * dr.t * dr.zt / dr.z - dr.S) / (dr.z * dr._ ** 2)
#             )   
#             - dr.gzz * dr.zz / dr.z
#         )
#     dr.divt = cth * dr.gtt / (1-cth**2) ** 0.5
    
#     # Relative divergence
#     with np.errstate(all='ignore'):
#         dr.divrelz = (
#             np.where(
#                 dr._ == 0.0, np.nan, 
#                   (2 * dr._ + 2 * dr.t * dr.zt / dr.z - dr.S) 
#                 / (dr._ ** 2 + (1-cth**2) * dr.t ** 2)
#             ) * dr.z
#             - dr.zz / dr.z
#         )
#     dr.divrelt = cth / (1-cth**2) ** 0.5 * np.ones_like(dr._)
#     return dr

# def find_characteristics(map_n, z0, all_t0, solver='DOP853', tol=1e-10) :
#     # Find domain
#     i0 = np.argwhere(zeta >= z0)[0, 0]
    
#     # Metric terms computation
#     dr = find_metric_terms(map_n)
#     dr = add_advanced_metric_terms(dr)
#     map_l  = pl_project_2D(map_n[i0:], L)
#     rhs_l  = pl_project_2D(dr.gg[i0:], L, even=False)
    
#     # Differential equation dt_dz = f(z, t)
#     def differential_equation(z, t) : 
#         # print(z, t)
#         rhs_t = pl_eval_2D(rhs_l, t)
#         f = interpolate_func(zeta[i0:], rhs_t, k=KSPL)(z) 
#         return f
    
#     # Actual solving
#     from scipy.integrate import solve_ivp
    
#     find_sol = lambda t :  solve_ivp(
#         fun=differential_equation, 
#         t_span=(z0, 1.0), 
#         y0=(t, ), 
#         method=solver, 
#         dense_output=True, 
#         rtol=tol, 
#         atol=tol
#     ).sol(zeta[i0:])[0]
    
#     all_sols = list(map(find_sol, all_t0))
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(15, 8.4), frameon=False)
#     sth = (1-cth**2)**0.5
#     x0_long = np.hstack(
#         ((map_n[i0]*sth)[::-1], -map_n[i0]*sth, (map_n[i0]*sth)[-1])
#     )
#     y0_long = np.hstack(
#         ((map_n[i0]*cth)[::-1],  map_n[i0]*cth, (map_n[i0]*cth)[-1])
#     )
#     ax.plot(x0_long, y0_long, lw=1.0, color='k')
#     for t in all_sols :
#         t = np.where(t > 1.0, 1.0, t)
#         r = np.array([pl_eval_2D(map_l[i], ti) for i, ti in enumerate(t)])
#         for sy in [1, -1] : 
#             for sx in [1, -1] :
#                 ax.plot(
#                     sx*r*(1-t**2)**0.5, sy*r*t, 
#                     lw=0.5, alpha=0.5, color='k', zorder=10
#                 )
#     plot_f_map(map_n, rho_n, phi_eff, L, add_to_fig=(fig, ax))
#     plt.show()
#     return all_sols

# def find_radiative_flux(map_n, z0, lum) :
#     """
#     Determination of the radiative flux from given isopotentials
#     (contained in map_n). This flux is determined by solving
#     div(Q) = 0 on each degree of the harmonic decomposition
#     and assuming that Q is normal to any level surface.

#     Parameters
#     ----------
#     map_n : array_like, shape (N, M)
#         Current mapping.
#     z0 : float
#         Zeta value from which div(Q) = 0 is solved.
#     lum : float
#         Integrated luminosity on surface.

#     Raises
#     ------
#     ValueError
#         If the matrix inversion enconters a difficulty ...

#     Returns
#     -------
#     Q_l : array_like, shape (N, L), optional
#         Radiative flux harmonics.

#     """    
#     # Solving domain
#     N0 = np.argwhere(zeta >= z0)[0, 0]
    
#     # Empty harmonics initialisation
#     l = np.arange(0, L, 2)
#     Q_l = np.zeros((N, L))
    
#     # Metric terms and coupling integral computation
#     dr = find_metric_terms(map_n, z0=z0)
#     Pll = DotDict()
#     Pll.zz = Legendre_coupling(
#         (dr._**2 + (1-cth**2) * dr.t**2) / dr.z, L, der=(0, 0)
#     )
#     Pll.zt = Legendre_coupling(
#         (1-cth**2) * dr.tt - 2*cth * dr.t, L, der=(0, 0)
#     )      + Legendre_coupling(
#         (1-cth**2) * dr.t, L, der=(0, 1)
#     )
    
#     # Vector and band matrix storage
#     Nl = (L+1)//2               
#     kl = (KLAG + 1) * Nl - 1      
#     ku = (KLAG + 0) * Nl - 1
#     b  = np.zeros(((N-N0)*Nl, ))
#     b[0::Nl] = rho_n[N0:]
#     blocs = np.empty((2*KLAG*Nl, (N-N0)*Nl))
#     ab = np.zeros((2*kl+ku+1, (N-N0)*Nl))
    
#     # Interpolation / derivation matrices
#     lag_mat = lagrange_matrix_P(zeta[N0:], order=KLAG)
#     Lsp = sps.dia_matrix(lag_mat[..., 0])
#     Dsp = sps.dia_matrix(lag_mat[..., 1])
#     Lsp_d_broad = Lsp.data[::-1, :, None, None]
#     Dsp_d_broad = Dsp.data[::-1, :, None, None]
        
#     # Main matrix parts filling
#     temp  = Dsp_d_broad * Pll.zz - Lsp_d_broad * Pll.zt
#     blocs = np.moveaxis(temp, 2, 1).reshape((2*KLAG*Nl, (N-N0)*Nl))
        
#     # Inner boundary conditions 
#     blocs[ku-Nl+1:ku+1, :Nl] = np.diag((1, ) + (0, )*(Nl-1))
    
#     # Matrix reindexing (credits to N. Fargette for this part)
#     mask = np.zeros(((N-N0)*Nl, kl+ku+1), dtype=bool)
#     for l in range(Nl) : 
#         mask[l::Nl, Nl-1-l:kl+ku+1-l] = 1
#     (ab[kl:, :]).T[mask] = (blocs.T).flatten()
                
#     # Matrix inversion (LAPACK)
#     # from scipy.sparse import diags
#     # from scipy.sparse.linalg import eigsh, lobpcg, norm, inv
#     # diag = tuple(ab[kl:][::-1])
#     # ab_sps = diags(diag, offsets=np.arange(len(diag))-kl, shape=((N-N0)*Nl, (N-N0)*Nl))
#     # v, w = eigsh(ab_sps, k=1, which='SM')
#     from scipy.linalg.lapack import dsbev
#     w, z, info = dsbev(ab, overwrite_ab=0)
#     _, _, x, info = dgbsv(kl, ku, ab, b)

#     if info != 0 : 
#         raise ValueError(
#             "Problem in the matrix factorisation. \n",
#             "Info = ", info
#         )
            
#     # Poisson's equation solution
#     Q_l[N0:, ::2] = x.reshape((N-N0, Nl))
#     Q = pl_eval_2D(Q_l, cth)
#     plot_f_map(map_n, Q, phi_eff, L, cmap='hot')
#     return Q_l

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