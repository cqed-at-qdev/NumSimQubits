# fluxonium.py
#
###################################################
### This file is part of numsimqubits.          ###
###                                             ###    
### Copyright (c) 2020 and later, Andras Gyenis ###
### All rights reserved.                        ###
###################################################

import numpy as np
from scipy.sparse import kron
import numsimqubits.qubits.build_operators as bop
import numsimqubits.qubits.settings as sim_settings
from numsimqubits.qubits.sccircuit import SCCircuit

class Zeropi(SCCircuit):
    """A class derived from SCCircuit for representing the zeropi qubit.
    
    Args:
        E_J (float): Josephson energy [GHz].
        E_C_theta (float): Charging energy of the theta mode [GHz].
        E_C_phi (float): Charging energy of the phi mode [GHz].
        E_L (float): Inductive energy [GHz].
        n_gate (float): Offset charge [pm 0.5 corresponds to the charge degeneracy point].
        n_gate_ls (ndarray): Offset charge values where the qubit properties are evaluated.
        phi_ext (float): External flux [pm 0.5 corresponds to the half flux quantum].
        phi_ext_ls (ndarray): External flux values where the qubit properties are evaluated.
        phase_limits (list(float)): The boundary of the phase space where the fluxonium is defined.
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].
        
    Attributes:
        E_J (float): Josephson energy [GHz].
        E_C_theta (float): Charging energy of the theta mode [GHz].
        E_C_phi (float): Charging energy of the phi mode [GHz].
        E_L (float): Inductive energy [GHz].
        n_gate (float): Offset charge [pm 0.5 corresponds to the charge degeneracy point].
        n_gate_ls (ndarray): Offset charge values where the qubit properties are evaluated.
        phi_ext (float): External flux [pm 0.5 corresponds to the half flux quantum].
        phi_ext_ls (ndarray): External flux values where the qubit properties are evaluated.
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
        n_theta (ndarray): Charge matrix elements for the theta mode at the different external offset charge values.
        n_phi (ndarray): Charge matrix elements for the phi mode at the different external offset charge values.
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
        fourier_operator(): Returns the Fourier operator to calculate the wavefunctions in charge basis from the wavefunctions in phase basis. 
    
    """

    def __init__(self, E_J, E_C_theta, E_C_phi, E_L, n_gate, n_gate_ls, phi_ext, phi_ext_ls, keig, dims, phase_limits, beta, f_resonator):
        """
        Zeropi constructor.
        """
        
        self.E_J               = E_J
        self.E_C_theta         = E_C_theta
        self.E_C_phi           = E_C_phi
        self.E_L               = E_L
        self.n_gate            = n_gate
        self.n_gate_ls         = n_gate_ls
        self.phi_ext           = phi_ext
        self.phi_ext_ls        = phi_ext_ls
        self.phase_limits      = phase_limits
        self.charge_basis      = [False, True] # [phi, theta]
        self.compact_variables = [False, True] # [phi, theta]
        self.operator_names    = ['n_theta', 'n_phi', 'phi']

        super(Zeropi, self).__init__(keig, dims, self.charge_basis, self.compact_variables, phase_limits, beta, f_resonator) # inherit attributes from base class

 
    def __repr__(self):
        """
        Gives complete information on the Zeropi object.
        """
        return 'Fluxonium(E_J=%r, E_C_theta=%r, E_C_phi=%r, E_L=%r, phi_ext=%r, n_gate=%r, phase_limits=%r, dims=%r, keig=%r, beta=%r, f_resonator=%r)' % (
            self.E_J, self.E_C_theta, self.E_C_phi, self.E_L, self.phi_ext, self.n_gate, self.phase_limits, self.dims, self.keig, self.beta, self.f_resonator)

    def hamiltonian(self):
        """Constructs the zeropi Hamiltonian, potential, and system operators in the proper basis.
        
        Returns:
            ndarray: Hamiltonian, potential energy and system operators.
        """

        # Load operators for phi in phase basis.
        ops_phi = bop.operators_phase_basis(compact_variable=self.compact_variable[0], Nx=self.dims[0], Dx=self.phase_limits[0]) 

        # Load operators for theta in charge basis.
        ops_theta = bop.operators_charge_basis(Nx=self.dims[1])

        # Build Hamiltonian (kinetic and potential terms).
        K = (- 4.0 * self.E_C_phi   * kron(ops_phi['d2x'],  ops_theta['id']) +
             + 4.0 * self.E_C_theta * kron(ops_phi['id'],   ops_theta['x2']) +
             - 8.0 * self.E_C_theta * kron(ops_phi['id'],   ops_theta['x']) * self.n_gate)
        V = (- 2.0 * self.E_J *       kron(ops_phi['cosx'], ops_theta['cosx']) * np.cos(2 * np.pi * self.phi_ext / 2) +
             - 2.0 * self.E_J *       kron(ops_phi['sinx'], ops_theta['cosx']) * np.sin(2 * np.pi * self.phi_ext / 2) +
             + 1.0 * self.E_L *       kron(ops_phi['x2'],   ops_theta['id']) )

        H = K + V

        # Build system operators.
        system_operators = {}

        system_operators['n_theta']  = kron(ops_phi['id'],   ops_theta['x'])
        system_operators['n_phi']    = kron(ops_phi['d1x'],  ops_theta['id'])
        system_operators['phi']      = kron(ops_phi['x'],    ops_theta['id'])
        system_operators['g_matrix'] = sim_settings.BETA_0 * self.f_resonator * (self.beta[0] * system_operators['n_phi'] + self.beta[1] * system_operators['n_theta'])

        return H, V, system_operators

    def fourier_operator(self):
        """
        Returns the Fourier operator to calculate the wavefunctions in charge basis from the wavefunctions in phase basis.
        """

        ops_n   = bop.operators_charge_basis(Nx=self.dims[1])
        ops_phi = bop.operators_phase_basis(compact_variable=True, Nx=self.dims[1])

        n_pts   = ops_n['x'].diagonal() 
        phi_pts = ops_phi['x'].diagonal() 
     
        fft_operator = np.array([[np.exp(-1j*phi*n) / np.sqrt(self.dims[1]) 
                                for phi in phi_pts] 
                                for n in n_pts])

        return fft_operator

    