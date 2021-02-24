# settings.py
#
###################################################
### This file is part of numsimqubits.          ###
###                                             ###    
### Copyright (c) 2020 and later, Andras Gyenis ###
### All rights reserved.                        ###
###################################################

import matplotlib as mpl
import numpy as np

# General constants     
Q_E = 1.602e-19 # electron charge in C
H   = 6.626e-34 # Planck constant in Js
Z_0 = 376.730   # impedance of free space in Ohm     
Z_R = 50.       # impedance of the resonator

H_BAR  = H /  (2. * np.pi)                     # Planck constant / 2p in Js
PHI_0  = H_BAR / (2. * Q_E)                    # flux quantum / 2p in Wb
ALPHA  = Q_E ** 2. * Z_0 / 4. / np.pi / H_BAR  #  fine_structure constant
BETA_0 = 4. * np.sqrt(ALPHA * Z_R / Z_0)       # coupling coefficient


# Simulation parameters
N_EIG_TOTAL  = 20           # number of eigenstates to calculate
DIAG_TOL     = 1.e-10       # diagonalization tolerance
DTYPE        = 'float32' # data type for vectors, matrices
MFORMAT      = 'csr'        # format of sparse matrices
NFOCK        = 3            # number of Fock states for coupled resonator + qubit systems

def update():

    # figure properties
    font_size         = 9
    line_width        = 0.5
    major_tick_length = 2 
    minor_tick_length = 1
    axes_color        = [0, 0, 0, 1]  

    params = {
              'text.usetex'         : True,
              'font.size'           : font_size,
              'legend.fontsize'     : font_size,
              'axes.labelsize'      : font_size,         
              'axes.linewidth'      : line_width,
              'axes.labelcolor'     : axes_color,
              'axes.titlesize'      : font_size,
              'axes.titlepad'       : 2,
              'legend.frameon'      : False,
              'legend.handletextpad': 0.2,
              'legend.handlelength' : 1.0,
              'legend.labelspacing' : 0.2,
              # x ticks
              'xtick.color'        : axes_color,
              'xtick.direction'    : 'in',
              'xtick.labelsize'    : font_size,          
              'xtick.major.width'  : line_width,
              'xtick.minor.width'  : line_width,
              'xtick.major.size'   : major_tick_length,
              'xtick.minor.size'   : minor_tick_length,
              'xtick.bottom'       : True,
              'xtick.major.bottom' : True,
              'xtick.minor.bottom' : True,
              'xtick.top'          : True,
              'xtick.major.top'    : True,
              'xtick.minor.top'    : True,
               # y ticks
              'ytick.color'        : axes_color,
              'ytick.direction'    : 'in',
              'ytick.labelsize'    : font_size,          
              'ytick.major.width'  : line_width,
              'ytick.minor.width'  : line_width,
              'ytick.major.size'   : major_tick_length,
              'ytick.minor.size'   : minor_tick_length,
              'ytick.left'         : True,
              'ytick.major.left'   : True,
              'ytick.minor.left'   : True,
              'ytick.right'        : True,
              'ytick.major.right'  : True,
              'ytick.minor.right'  : True,          
              }



    mpl.rcParams.update(params)
    
    
