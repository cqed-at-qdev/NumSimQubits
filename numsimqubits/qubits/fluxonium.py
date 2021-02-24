# fluxonium.py
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

class Fluxonium(SCCircuit):
    """A class derived from SCCircuit for representing the fluxonium qubit.
    
    Args:
        E_J (float): Josephson energy [GHz].
        E_C (float): Charging energy [GHz].
        E_L (float): Inductive energy [GHz].
        phi_ext (float): External flux [pm 0.5 corresponds to the half flux quantum].
        phi_ext_ls (ndarray): External flux values where the qubit properties are evaluated.
        phase_limits (list(float)): The boundary of the phase space where the fluxonium is defined.
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].
        
    Attributes:
        E_J (float): Josephson energy [GHz].
        E_C (float): Charging energy [GHz].
        E_L (float): Inductive energy [GHz].
        phi_ext (float): External flux [pm 0.5 corresponds to the half flux quantum].
        phi_ext_ls (list(float)): External flux values where the qubit properties are evaluated.
        phase_limits (list(float)): The boundary of the phase space where the fluxonium is defined.
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].
        compact_variables (list(bool)): The circuit's phase variables are compact or not, i.e., defined with periodic boundary conditions if True.
        charge_basis (list(bool)): The variables are treated in charge basis if True, in phase basis if False.
        operator_names (list(str)): The names of relevant system operators for the qubit.
        axis (list(ndarray)): Points along the charge or phase axis for each degree of freedom.
        evals (ndarray): Eigenvalues of the circuit at the different external parameters [GHz].
        evals_full (ndarray): Eigenvalues of the coupled circuit and resonator at the different external parameters [GHz].
        ekets (ndarray): Eigenstates of the circuit at the different offset charge values. 
        n (ndarray): Charge matrix elements at the different external offset charge values.
        phi (ndarray): Phase matrix element at the different external parameters. 
        g (ndarray): Coupling rates at the different external offset charge values [GHz].
        chi (ndarray): Dispersive shifts at the different external offset charge values [GHz].
        kappa (ndarray): Lamb shifts at the different external offset charge values [GHz].
        g (ndarray): Coupling rates at the different external offset charge values [GHz].
        chi (ndarray): Dispersive shifts at the different external offset charge values [GHz].
        kappa (ndarray): Lamb shifts at the different external offset charge values [GHz].

    Methods:
        diagonalize(): Diagonalizes the Hamiltonian of the circuit, calculates matrix elements, coupling rates, dispersive shifts.
        sweep_diagonalize(): Diagonalizes the Hamiltonian and updates the attributes of the qubit at the parameters corresponding to the sweep variable.
        sweep_diagonalize_full_Hamiltonian(): Diagonalizes the coupled circuit and resonator Hamiltonian.
        extract_results(): Converts the list obtained from the parameter sweep to more convenient format.
        hamiltonian(): Constructs the transmon Hamiltonian, potential, and system operators in the proper basis.
    
    """

    def __init__(self, E_J, E_C, E_L, phi_ext, phi_ext_ls, phase_limits, dims, keig, beta, f_resonator):
        """
        Fluxonium constructor.
        """
        
        self.E_J               = E_J
        self.E_C               = E_C
        self.E_L               = E_L
        self.phi_ext           = phi_ext
        self.phi_ext_ls        = phi_ext_ls
        self.phase_limits      = phase_limits
        self.charge_basis      = [False]
        self.compact_variables = [False]
        self.operator_names    = ['n', 'phi']

        super(Fluxonium, self).__init__(keig, dims, self.charge_basis, self.compact_variables, phase_limits, beta, f_resonator) # inherit attributes from base class

 
    def __repr__(self):
        """
        Gives complete information on the Fluxonium object.
        """
        return 'Fluxonium(E_J=%r, E_C=%r, E_L=%r, phi_ext=%r, phase_limits=%r, dims=%r, keig=%r, charge_basis=%r, beta=%r, f_resonator=%r)' % (
            self.E_J, self.E_C, self.E_L, self.phi_ext, self.phase_limits, self.dims, self.keig, self.charge_basis, self.beta, self.f_resonator)

    def hamiltonian(self):
        """Constructs the transmon Hamiltonian, potential, and system operators in the proper basis.
        
        Returns:
            ndarray: Hamiltonian, potential energy and system operators.
        """

        # load operators
        ops = bop.operators_phase_basis(compact_variable=self.compact_variables[0], Nx=self.dims[0], Dx=self.phase_limits[0])

        # Build Hamiltonian (kinetic and potential terms)
        K = (- 4.0 * self.E_C * ops['d2x'])
        V = (- 1.0 * self.E_J * ops['cosx'] * np.cos(self.phi_ext*2*np.pi) +
             - 1.0 * self.E_J * ops['sinx'] * np.sin(self.phi_ext*2*np.pi) +
             + 0.5 * self.E_L * ops['x2'])

        H = K + V

        # Build system operators.
        system_operators = {}

        system_operators['n']        = -1j * ops['d1x']
        system_operators['phi']      = ops['x']
        system_operators['g_matrix'] = sim_settings.BETA_0 * self.f_resonator * self.beta[0] * system_operators['n']

        return H, V, system_operators

