# transmon.py
#
###################################################
### This file is part of numsimqubits.          ###
###                                             ###    
### Copyright (c) 2020 and later, Andras Gyenis ###
### All rights reserved.                        ###
###################################################

import numpy as np
import numsimqubits.qubits.build_operators as bop
import numsimqubits.qubits.settings as sim_settings
from numsimqubits.qubits.sccircuit import SCCircuit

class Transmon(SCCircuit):
    """A class derived from SCCircuit for representing the transmon qubit.
    
    Args:
        E_J (float): Josephson energy [GHz].
        E_C (float): Charging energy [GHz].
        n_gate (float): Offset charge [pm 0.5 corresponds to the charge degeneracy point].
        n_gate_ls (list(float)): Offset charge values where the qubit properties are evaluated.
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        charge_basis (list(bool)): The variables are treated in charge basis if True, in phase basis if False.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].
        
    Attributes:
        E_J (float): Josephson energy [GHz].
        E_C (float): Charging energy [GHz].
        n_gate (float): Offset charge [pm 0.5 corresponds to the charge degeneracy point].
        n_gate_ls (ndarray): Offset charge values where the qubit properties are evaluated.
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        charge_basis (list(bool)): The variables are treated in charge basis if True, in phase basis if False.
        compact_variables (list(bool)): The circuit's phase variables are compact or not, i.e., defined with periodic boundary conditions if True.
        phase_limits (list(float)): The boundaries of the phase space where the cricuit Hamiltonian is defined.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].
        operator_names (list(str)): The names of relevant system operators for the qubit.
        axis (list(ndarray)): Points along the charge or phase axis for each degree of freedom.
        evals (ndarray): Eigenvalues of the circuit at the different external parameters [GHz].
        evals_full (ndarray): Eigenvalues of the coupled circuit and resonator at the different external parameters [GHz].
        ekets (ndarray): Eigenstates of the circuit at the different offset charge values. 
        n (ndarray): Charge matrix elements at the different external offset charge values.
        g (ndarray): Coupling rates at the different external offset charge values [GHz].
        chi (ndarray): Dispersive shifts at the different external offset charge values [GHz].
        kappa (ndarray): Lamb shifts at the different external offset charge values [GHz].
        
    Methods:
        diagonalize(): Diagonalizes the Hamiltonian of the circuit, calculates matrix elements, coupling rates, dispersive shifts.
        sweep_diagonalize(): Diagonalizes the Hamiltonian and updates the attributes of the qubit at the parameters corresponding to the sweep variable.
        sweep_diagonalize_full_Hamiltonian(): Diagonalizes the coupled circuit and resonator Hamiltonian.
        extract_results(): Converts the list obtained from the parameter sweep to more convenient format.
        hamiltonian(): Constructs the transmon Hamiltonian, potential, and system operators in the proper basis.
        fourier_operator(): Returns the Fourier operator to calculate the wavefunctions in charge basis from the wavefunctions in phase basis. 
    
    """

    def __init__(self, E_J, E_C, n_gate, n_gate_ls, dims, keig, charge_basis, beta, f_resonator):
        """
        Transmon constructor.
        """
        
        self.E_J               = E_J
        self.E_C               = E_C
        self.n_gate            = n_gate
        self.n_gate_ls         = n_gate_ls
        self.phase_limits      = [np.pi]
        self.charge_basis      = charge_basis 
        self.compact_variables = [True]
        self.operator_names    = ['n']

        super(Transmon, self).__init__(keig, dims, charge_basis, self.compact_variables, self.phase_limits, beta, f_resonator) # inherit attributes from base class

    def __repr__(self):
        """
        Gives complete information on the Transmon object.
        """
        return 'Transmon(E_J=%r, E_C=%r, n_gate=%r, dims=%r, keig=%r, charge_basis=%r, beta=%r, f_resonator=%r)' % (
            self.E_J, self.E_C, self.n_gate, self.dims, self.keig, self.charge_basis, self.beta, self.f_resonator)

    def hamiltonian(self):
        """Constructs the transmon Hamiltonian, potential, and system operators in the proper basis.
        
        Returns:
            ndarray: Hamiltonian, potential energy and system operators.
        """

        if self.charge_basis[0]:

            # Load operators in charge basis.
            ops = bop.operators_charge_basis(Nx=self.dims[0])

            # Build Hamiltonian (kinetic and potential terms).
            K = - 1.0 * self.E_J * ops['cosx']
            V = + 4.0 * self.E_C * ops['x2'] - 8.0 * self.E_C * ops['x'] * self.n_gate

            H = K + V

            # Build system operators.
            system_operators = {}

            system_operators['n']  = ops['x']
            system_operators['g_matrix'] = sim_settings.BETA_0 * self.f_resonator * self.beta[0] * system_operators['n']
     
            return H, V, system_operators

        else:

            # Load operators in phase basis.
            ops = bop.operators_phase_basis(compact_variable=self.compact_variables[0], Nx=self.dims[0])

            # Build Hamiltonian (kinetic and potential terms).
            K = - 4.0 * self.E_C * ops['d2x'] + 8.0 * self.E_C * 1j * ops['d1x'] * self.n_gate
            V = - 1.0 * self.E_J * ops['cosx']

            H = K + V

            # Build system operators.
            system_operators = {}

            system_operators['n']  = -1j * ops['d1x']
            system_operators['g_matrix'] = sim_settings.BETA_0 * self.f_resonator * self.beta[0] * system_operators['n']
     
            return H, V, system_operators

    def fourier_operator(self):
        """
        Returns the Fourier operator to calculate the wavefunctions in charge basis from the wavefunctions in phase basis.
        """

        ops_n   = bop.operators_charge_basis(Nx=self.dims[0])
        ops_phi = bop.operators_phase_basis(compact_variable=True, Nx=self.dims[0])

        n_pts   = ops_n['x'].diagonal() 
        phi_pts = ops_phi['x'].diagonal() 
     
        fft_operator = np.array([[np.exp(-1j*phi*n) / np.sqrt(self.dims[0]) 
                                for phi in phi_pts] 
                                for n in n_pts])

        return fft_operator



   

    





