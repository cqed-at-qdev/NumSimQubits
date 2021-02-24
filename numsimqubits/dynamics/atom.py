# atom.py
#
###################################################
### This file is part of numsimqubits.          ###
###                                             ###    
### Copyright (c) 2020 and later, Andras Gyenis ###
### All rights reserved.                        ###
###################################################

import numpy as np
from qutip.interpolate import Cubic_Spline
import qutip as qt

class Atom:
    """ Class for representing an atom with discrete energy levels.

    Args:
        N_levels (int): The number of energy levels.
        energies (list(float)): The energy of the atomic levels [GHz].
        couplings (list(float)): The couplings between the levels. 
        depolar_rates (ndarray): The decay between energy levels [GHz].
        dephase_rates (ndarray): The dephasing between energy levels [GHz].
        initial_state (Qobj): The initial state ot the atom.

    Attributes:
        N_levels (int): The number of energy levels.
        energies (list(float)): The energy of the atomic levels [GHz].
        couplings (list(float)): The couplings between the levels.  
        depolar_rates (ndarray): The decay between energy levels [GHz].
        dephase_rates (ndarray): The dephasing between energy levels [GHz].
        initial_state (Qobj): The initial state ot the atom.
        population (list): The atomic populations as a function of time.
        density_matrices (list(Qobj)): The atomic density matrices as a function of time.
        
    Methods:
        projection_operators(): Defines projection operators.
        collapse_operators(): Defines collapse operators.
        get_population(): Calculates the atomic populations.
        calculate_dynamics(): Calculates the time evolution of the system.

    """

    def __init__(self, N_levels, energies, couplings, depolar_rates, dephase_rates, initial_state):
        """
        Gaussianpulse constructor.
        """

        self.N_levels      = N_levels
        self.energies      = energies
        self.couplings     = couplings
        self.depolar_rates = depolar_rates
        self.dephase_rates = dephase_rates
        self.initial_state = initial_state
    
    def __repr__(self):
        """
        Gives complete information on the Gaussianpulse object.
        """
        
        return 'Atom(\n N_levels=%r, \n energies=%r, \n couplings=%r, \n depolar_rates=%r, \n dephase_rates=%r, \n initial_state=%r \n)' % (
           self.N_levels, self.energy, self.couplings, self.depolar_rate, self.dephase_rate, self.initial_state)

    def projection_operators(self):
        """
        Defines projection operators.
        """

        s = [[qt.projection(self.N_levels,i,j) 
                            for i in range(self.N_levels)] 
                            for j in range(self.N_levels)] 
        
        return s

    def collapse_operators(self):
        """
        Defines collapse operators.
        """

        s = self.projection_operators()

        c_ops = [[np.sqrt(self.depolar_rates[i][j] * 2 * np.pi * 1e9) * s[i][j]
                            for i in range(self.N_levels)] 
                            for j in range(self.N_levels)]

        c_ops.append([np.sqrt(self.dephase_rates[i][i] * 4 * np.pi * 1e9) * s[i][i]
                            for i in range(self.N_levels)] )

        return c_ops

    def get_population(self):
        """
        Calculates the atomic populations.
        """

        s = self.projection_operators()

        self.population = [[qt.expect(s[level_idx][level_idx], self.density_matrices[t]) 
                                for t in range(len(self.density_matrices))]
                                for level_idx in range(self.N_levels)]

        return

    def calculate_dynamics(self, system, *pulse):
        """
        Calculates the time evolution of the system.

        Args:
            system (func): function that returns the Hamiltonian of the coupled atom - pulse system.
            pulse (object): object that represents a pulse.

        """

        operator = system(self, *pulse) # System Hamiltonian.
        s = self.projection_operators() # Sigma operators.
        
        results = qt.mesolve(operator, self.initial_state, pulse[0].t_ls, c_ops=self.collapse_operators())

        self.density_matrices = results.states

        self.population = [[qt.expect(s[level_idx][level_idx], self.density_matrices[t]) 
                                for t in range(len(self.density_matrices))]
                                for level_idx in range(self.N_levels)]
       
        return

