# build_operators.py
#
###################################################
### This file is part of numsimqubits.          ###
###                                             ###    
### Copyright (c) 2020 and later, Andras Gyenis ###
### All rights reserved.                        ###
###################################################


import numpy as np
from scipy.sparse import diags

import numsimqubits.qubits.settings as sim_settings

def operators_charge_basis(Nx, T=[], phi_ext=0.0, dtype=sim_settings.DTYPE, mformat=sim_settings.MFORMAT):
    """
    Returns sparse operators in charge basis. 

    Args:
        Nx (int): Number of grid points along the charge axis.
        T (list(float)): Transmission coefficients for a single channel to calculate the potential for an Andreev bound state.
        phi_ext (float) : External flux.
        dtype (str): Data type for vectors, matrices.
        mformat (str): Format for sparse matrices

    Returns:
        dict(sparse): Basic operators in sparse matrix format.

    """

    operators = {}

    x_pts = np.linspace(-(Nx-1)/2, (Nx-1)/2, Nx, endpoint=True, dtype=dtype)  

    operators['id']   = diags(np.ones(x_pts.size), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['x']    = diags(x_pts, 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['x2']   = diags(x_pts**2, 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['cosx'] = diags([0.5,0.5], [-1,1], shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['sinx'] = 1j * diags([-0.5,0.5], [-1,1], shape=(Nx,Nx), format=mformat, dtype=dtype)

    # Calculate the operator for the potential of an Andreev bound state.
    operators['ABS']  = diags(np.zeros(x_pts.size), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    
    phase_pts = np.linspace(-np.pi, np.pi, Nx, endpoint=False, dtype=dtype)

    for T_i in T:

        Vx        = np.sqrt(1 - T_i * (1 - np.cos(phase_pts - 2 * np.pi * phi_ext)) / 2)
        V_FT_cos  = []
        V_FT_sin  = []

        for idx in range(Nx):
            V_FT_cos.append(np.sum(np.cos(idx * phase_pts) * Vx ) / Nx)
            V_FT_sin.append(np.sum(np.sin(idx * phase_pts) * Vx ) / Nx)
    
        for k in range(1,Nx):
            operators['ABS'] += V_FT_cos[k] * diags([1,1], [-k,k], shape=(Nx,Nx), format=mformat, dtype=dtype)
            operators['ABS'] += 1j* V_FT_sin[k] * diags([-1,1], [-k,k], shape=(Nx,Nx), format=mformat, dtype=dtype)

    # Parity operator

    operators['parity'] = diags(np.array([(-1)**x for x in x_pts]), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)

    return operators

def operators_phase_basis(compact_variable, Nx, Dx=4*np.pi, T=[], phi_ext=0.0, dtype=sim_settings.DTYPE, mformat=sim_settings.MFORMAT):
    """
    Returns sparse operators in phase basis. 

    Args:
        compact_variable (bool): The operator is considered in a periodic or non-periodic space.
        Nx (int): Number of grid points along the phase axis.
        Dx (int): Bounderies of the phase axis.
        T (list(float)): Transmission coefficients for a single channel to calculate the potential for an Andreev bound state.
        phi_ext (float) : External flux.
        dtype (str): Data type for vectors, matrices.
        mformat (str): Format for sparse matrices

    Returns:
        dict(sparse): Basic operators in sparse matrix format.
        
    """ 

    if compact_variable:
        x_pts = np.linspace(-np.pi, np.pi, Nx, endpoint=False, dtype=dtype)

    else:
        x_pts = np.linspace(-Dx, Dx, Nx, endpoint=True, dtype=dtype)

    dx = x_pts[-1] - x_pts[-2] 

    d1_coeff  = (1. / (2. * dx))
    d2_coeff  = (1. / (dx ** 2))  

    operators = {}

    operators['id']   = diags(np.ones(x_pts.size), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['x']    = diags(x_pts, 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['x2']   = diags(x_pts**2, 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['cosx'] = diags(np.cos(x_pts), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['sinx'] = diags(np.sin(x_pts), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['d1x']  = diags([-d1_coeff, d1_coeff], [-1,1], shape=(Nx,Nx), format=mformat, dtype=dtype)
    operators['d2x']  = diags([d2_coeff, -2.0 * d2_coeff, d2_coeff], [-1,0,1], shape=(Nx,Nx), format=mformat, dtype=dtype)
    
    # Calculate the operator for the potential of an Andreev bound state.
    operators['ABS']  = diags(np.zeros(x_pts.size), 0, shape=(Nx,Nx), format=mformat, dtype=dtype)

    for T_i in T:
        Vx                = np.sqrt(1 - T_i * (1 - np.cos(x_pts - 2 * np.pi * phi_ext)) / 2)
        operators['ABS'] += diags(Vx, 0, shape=(Nx,Nx), format=mformat, dtype=dtype) 

    # boundary conditions for compact variables:
    if compact_variable:
        d1x_BC = diags([d1_coeff,-d1_coeff], [-(Nx-1),(Nx-1)], shape=(Nx,Nx), format=mformat, dtype=dtype)
        d2x_BC = diags([d2_coeff,d2_coeff], [-(Nx-1),(Nx-1)], shape=(Nx,Nx), format=mformat, dtype=dtype)
        operators['d1x'] += d1x_BC        
        operators['d2x'] += d2x_BC      
            
    return operators