# driven_systems.py
#
###################################################
### This file is part of numsimqubits.          ###
###                                             ###    
### Copyright (c) 2020 and later, Andras Gyenis ###
### All rights reserved.                        ###
###################################################


import numpy as np
import qutip as qt
import copy
from qutip.interpolate import Cubic_Spline

def combine_pulses(*pulses):
    """
    Combines multiple pulses into a single pulse.

    Args:
        pulse (object): The pulse shape. 

    Returns:
        obj: combined pulse.

    """

    pulse_all = copy.deepcopy(pulses[0])

    pulse_base_I = np.zeros(shape = len(pulse_all.t_ls)) 
    pulse_base_Q = np.zeros(shape = len(pulse_all.t_ls)) 

    for pulse in pulses:

        pulse_base_I += pulse.I(pulse_all.t_ls)
        pulse_base_Q += pulse.Q(pulse_all.t_ls)

    pulse_all.I = Cubic_Spline(pulse_all.t_ls[0], pulse_all.t_ls[-1], pulse_base_I)
    pulse_all.Q = Cubic_Spline(pulse_all.t_ls[0], pulse_all.t_ls[-1], pulse_base_Q)

    return pulse_all

     
def two_level_one_drive(atom, pulse):
    """
    Defines the time-depenent Hamiltonian for a two level atomic system coupled with one drive term.

    Args:
        atom (object): The atomic system.
        pulse (object): The pulse shape. 

    Returns:
        list: Hamiltonian as operator-function pairs.

    """

    # Extract the energy levels for the atom.
    E_0  = atom.energies[0] 
    E_1  = (atom.energies[1] - E_0) * 2 * np.pi * 1e9

    # Extract coupling rates between levels.
    n01 = atom.couplings[0][1]

    # Extract sigma operators.
    s = atom.projection_operators() 

    # Extract the drive frequency.
    freq = pulse.freq * 2 * np.pi  * 1e9

    # Time-independent part of the Hamiltonian.
    H_0 = (E_1 - freq) * s[1][1] 

    # Time-dependent real part of the Hamiltonian.
    H_I = [n01 * (s[0][1] + s[1][0]) / 2, pulse.I]

    # Time-dependent imaginary part of the Hamiltonian.
    H_Q = [n01 * 1j * (s[0][1] - s[1][0]) / 2, pulse.Q]

    H = [H_0, H_I, H_Q]
   
    return H

def three_level_two_drive(atom, pulse_alpha, pulse_beta):
    """
    Defines the time-depenent Hamiltonian for a three level lambda system coupled with two drive terms.

    Args:
        atom (object): The atomic system.
        pulse_alpha (object): The pulse_alpha shape.
        pulse_beta (object): The pulse_beta shape. 

    Returns:
        list: Hamiltonian as operator-function pairs.

    """
   
    # Extract the energy levels for the atom.
    E_0  = atom.energies[0] 
    E_1  = (atom.energies[1] - E_0) * 2 * np.pi * 1e9
    E_2  = (atom.energies[2] - E_0) * 2 * np.pi * 1e9  

    # Extract coupling rates between levels.
    n02 = atom.couplings[0][2]
    n12 = atom.couplings[1][2]

    # Extract sigma operators.
    s = atom.projection_operators()

    # Extract the drive frequencies.
    freq_alpha = pulse_alpha.freq * 2 * np.pi * 1e9 
    freq_beta  = pulse_beta.freq * 2 * np.pi * 1e9 

    # Time-independent part of the Hamiltonian.
    H_0 = (E_1 - freq_alpha + freq_beta) * s[1][1] + (E_2 - freq_alpha) * s[2][2] 

    # Time-dependent real part of the Hamiltonian.
    H_02_I = [n02 * (s[0][2] + s[2][0]) / 2, pulse_alpha.I]   
    H_12_I = [n12 * (s[1][2] + s[2][1]) / 2, pulse_beta.I]

    # Time-dependent imaginary part of the Hamiltonian.
    H_02_Q = [1j * (s[0][2] - s[2][0]) / 2, pulse_alpha.Q]
    H_12_Q  = [1j * (s[1][2] - s[2][1]) / 2, pulse_beta.Q]

    H = [H_0, H_02_I, H_12_I, H_02_Q, H_12_Q]
   
    return H

def four_level_two_drive(atom, pulse_alpha, pulse_beta):
    """
    Defines the time-depenent Hamiltonian for a four level system coupled with two drive terms.

    Args:
        atom (object): The atomic system.
        pulse_alpha (object): The pulse_alpha shape.
        pulse_beta (object): The pulse_beta shape. 

    Returns:
        list: Hamiltonian as operator-function pairs.

    """
    
    # Extract the energy levels for the atom.
    E_0  = atom.energies[0] 
    E_1  = (atom.energies[1] - E_0) * 2 * np.pi * 1e9
    E_2  = (atom.energies[2] - E_0) * 2 * np.pi * 1e9 
    E_3  = (atom.energies[3] - E_0) * 2 * np.pi * 1e9  

    # Extract coupling rates between levels.
    n03 = atom.couplings[0][3]
    n23 = atom.couplings[2][3]
    n13 = atom.couplings[1][3]

    # Extract sigma operators.
    s = atom.projection_operators()

    # Extract the drive frequencies.
    freq_alpha = pulse_alpha.freq * 2 * np.pi * 1e9 
    freq_beta  = pulse_beta.freq * 2 * np.pi * 1e9

    # Time-independent part of the Hamiltonian.
    H_0 = (E_1 - freq_alpha + freq_beta) * s[1][1] + (E_2 - freq_alpha + freq_beta) * s[2][2] + (E_3 - freq_alpha) * s[3][3]

    # Time-dependent real part of the Hamiltonian.
    H_03_I = [n03 * (s[0][3] + s[3][0]) / 2, pulse_alpha.I]
    H_23_I = [n23 * (s[2][3] + s[3][2]) / 2, pulse_beta.I]
    H_13_I = [n13 * (s[1][3] + s[3][1]) / 2, pulse_beta.I]

    # Time-dependent imaginary part of the Hamiltonian.
    H_03_Q = [n03 * 1j * (s[0][3] - s[3][0]) / 2, pulse_alpha.Q]
    H_23_Q = [n23 * 1j * (s[2][3] - s[3][2]) / 2, pulse_beta.Q]
    H_13_Q = [n13 * 1j * (s[1][3] - s[3][1]) / 2, pulse_beta.Q]

    H = [H_0, H_03_I, H_23_I, H_13_I, H_03_Q, H_23_Q, H_13_Q]
   
    return H

def three_level_one_drive_A(atom, pulse):
    """
    Defines the time-depenent Hamiltonian for a three level lambda system coupled with one drive term.
    The drive couples levels (0,1) and (0,2)

    Args:
        atom (object): The atomic system.
        pulse (object): The pulse shape.

    Returns:
        list: Hamiltonian as operator-function pairs.

    """

    # Extract the energy levels for the atom.
    E_0  = atom.energies[0] 
    E_1  = (atom.energies[1] - E_0) * 2 * np.pi * 1e9
    E_2  = (atom.energies[2] - E_0) * 2 * np.pi * 1e9

    # Extract coupling rates between levels.
    n01 = atom.couplings[0][1]
    n02 = atom.couplings[0][2]
    
    # Extract sigma operators.
    s = atom.projection_operators()

    # Extract the drive frequency.
    freq = pulse.freq * 2 * np.pi * 1e9 

    # Time-independent part of the Hamiltonian.
    H_0 = (E_1 - freq) * s[1][1] + (E_2 - freq) * s[2][2] 

    # Time-dependent real part of the Hamiltonian.
    H_01_I = [n01 * (s[0][1] + s[1][0]) / 2, pulse.I]
    H_02_I = [n02 * (s[0][2] + s[2][0]) / 2, pulse.I]
    
    # Time-dependent imaginary part of the Hamiltonian.
    H_01_Q = [n01 * 1j * (s[0][1] - s[1][0]) / 2, pulse.Q]
    H_02_Q = [n02 * 1j * (s[0][2] - s[2][0]) / 2, pulse.Q]

    H = [H_0, H_01_I, H_02_I, H_01_Q, H_02_Q]
   
    return H

def three_level_one_drive_B(atom, pulse):
    """
    Defines the time-depenent Hamiltonian for a three level lambda system coupled with one drive term.
    The drive couples levels (0,1) and (1,2)

    Args:
        atom (object): The atomic system.
        pulse (object): The pulse shape.

    Returns:
        list: Hamiltonian as operator-function pairs.

    """

    # Extract the energy levels for the atom.
    E_0  = atom.energies[0] 
    E_1  = (atom.energies[1] - E_0) * 2 * np.pi * 1e9
    E_2  = (atom.energies[2] - E_0) * 2 * np.pi * 1e9

    # Extract coupling rates between levels.
    n01 = atom.couplings[0][1]
    n12 = atom.couplings[1][2]
    
    # Extract sigma operators.
    s = atom.projection_operators()

    # Extract the drive frequency.
    freq = pulse.freq * 2 * np.pi * 1e9 

    # Time-independent part of the Hamiltonian.
    H_0 = (E_1 - freq) * s[1][1] + (E_2 - 2 * freq) * s[2][2] 

    # Time-dependent real part of the Hamiltonian.
    H_01_I = [n01 * (s[0][1] + s[1][0]) / 2, pulse.I]
    H_12_I = [n12 * (s[1][2] + s[2][1]) / 2, pulse.I]
    
    # Time-dependent imaginary part of the Hamiltonian.
    H_01_Q = [n01 * 1j * (s[0][1] - s[1][0]) / 2, pulse.Q]
    H_12_Q = [n12 * 1j * (s[1][2] - s[2][1]) / 2, pulse.Q]

    H = [H_0, H_01_I, H_12_I, H_01_Q, H_12_Q]
   
    return H



# def three_level_one_drive_B(t, atom, pulse):

#     E_0  = atom.energies[0] 
#     E_1  = (atom.energies[1] - E_0) * 2 * np.pi
#     E_2  = (atom.energies[2] - E_0) * 2 * np.pi

#     freq = pulse.freq * 2 * np.pi

#     s = atom.projection_operators()

#     pulse_I = pulse.base(t)[0]
#     pulse_Q = pulse.base(t)[1]

#     # Time-independent part of the Hamiltonian.
#     H_0 = (E_1 - freq) * s[1][1] + (E_2 - 2 * freq) * s[2][2] 

#     # Time-dependent real part of the Hamiltonian.
#     H_I1 = [(s[0][1] + s[1][0]) / 2, pulse_I]
#     H_I2 = [(s[1][2] + s[2][1]) / 2, pulse_I]

#     # Time-dependent imaginary part of the Hamiltonian.
#     H_Q1 = [1j * (s[0][1] - s[1][0]) / 2, pulse_Q]
#     H_Q2 = [1j * (s[1][2] - s[2][1]) / 2, pulse_Q]

#     H = [H_0, H_I1, H_I2, H_Q1, H_Q2]
   
#     return H

