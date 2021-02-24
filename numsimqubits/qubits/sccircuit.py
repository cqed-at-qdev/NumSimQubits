# sccircuit.py
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
from tqdm import tqdm
from scipy.sparse import linalg, diags, eye
from qutip import Qobj, qeye, tensor, destroy

class SCCircuit:
    """ A base class for representing a general superconducting circuit, such as transmon, fluxonium, zeropi.

    Args:
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        charge_basis (list(bool)): The variables are treated in charge basis if True, in phase basis if False.
        compact_variables (list(bool)): The circuit's phase variables are compact or not, i.e., defined with periodic boundary conditions if True.
        phase_limits (list(float)): The boundaries of the phase space where the cricuit Hamiltonian is defined.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].

    Attributes:
        keig (int): Number of states to calculate.
        dims (list(int)): Number of points in phase / charge space for each degree of freedom. Must be an add number.
        charge_basis (list(bool)): The variables are treated in charge basis if True, in phase basis if False.
        compact_variables (list(bool)): The circuit's phase variables are compact or not, i.e., defined with periodic boundary conditions if True.
        phase_limits (list(float)): The boundaries of the phase space where the cricuit Hamiltonian is defined.
        beta (list(float)): Coupling constants for a capacitively coupled circuit-resonator system.
        f_resonator (float): Frequency of the coupled resonator [GHz].
        axis (list(ndarray)): Points along the charge or phase axis for each degree of freedom.
        evals (ndarray): Eigenvalues of the circuit at the different external parameters [GHz].
        evals_full (ndarray): Eigenvalues of the coupled circuit and resonator at the different external parameters [GHz].
        ekets (ndarray): Eigenstates of the circuit at the different external parameters. 
        V (ndarray): Potential energy as function of the phase or charge variable at the different external parameters [GHz]. 
        zero_energy(float): The energy of the lowest lying state [GHz].
        g (ndarray): Coupling rates at the different external parameters [GHz]. 
        chi (ndarray): Dispersive shifts at the different external parameters [GHz].
        kappa (ndarray): Lamb shifts at the different external parameters [GHz]. 

    Methods:
        diagonalize(): Diagonalizes the Hamiltonian of the circuit, calculates matrix elements, coupling rates, dispersive shifts.
        sweep_diagonalize(): Diagonalizes the Hamiltonian and updates the attributes of the qubit at the parameters corresponding to the sweep variable.
        sweep_diagonalize_full_Hamiltonian(): Diagonalizes the coupled circuit and resonator Hamiltonian.
        extract_results(): Converts the list obtained from the parameter sweep to more convenient format.

    """

    def __init__(self, keig, dims, charge_basis, compact_variables, phase_limits, beta, f_resonator):
        """
        SCCircuit constructor.
        """

        self.keig             = keig
        self.dims             = dims
        self.charge_basis     = charge_basis
        self.compact_variable = compact_variables
        self.phase_limits     = phase_limits
        self.beta             = beta
        self.f_resonator      = f_resonator
        self.axis             = []

        for dim, basis, compact, limit in zip(self.dims, self.charge_basis, self.compact_variable, self.phase_limits):

            if basis: 
                ops = bop.operators_charge_basis(Nx=dim) # load operators in charge basis
                self.axis.append(ops['x'].diagonal())

            else:
                ops = bop.operators_phase_basis(compact_variable=compact, Nx=dim, Dx=limit) # load operators in phase basis
                self.axis.append(ops['x'].diagonal())


    def diagonalize(self, calculate_mx_elements=False):
        """Diagonalizes the Hamiltonian of the circuit, calculates matrix elements, coupling rates, dispersive shifts.
        
        Args:
            calculate_mx_elements (bool): Calculate the matrix elements, dispersive shifts, etc. if True.

        Returns:
            dict: Eigenvalues, eigenstates, energy of the lowest lying state, potential of the Hamiltonian, 
                matrix elements, coupling rates, dispersive and Lamb shifts. 

        """
   
        # Make sure the Hamiltonian is up-to-date with the latest parameters.
        H, V, system_operators = self.hamiltonian() 

        # Shift the potential minimum to 0.
        V_min = V.diagonal().min()
        V     = V.diagonal() - V_min
        H     = H - V_min * eye(H.shape[0])

        # Calculate eigenenergies and eigenvectors.
        evals, ekets = linalg.eigsh(H, k=self.keig, which='SA', tol=sim_settings.DIAG_TOL)
        sort_idxs    = evals.argsort()
        evals        = np.sort(evals)
        zero_energy  = evals[0]
        evals        = evals - zero_energy # Define the 0 energy with respect to the lowest lying state.

        ekets    = [ekets[:,idx].reshape(self.dims) for idx in sort_idxs] # Sort and reshape eigenvectors.

        diag_result = dict(evals=evals, ekets=ekets, V=V, zero_energy=zero_energy)

        # Calculate matrix elements.
        if calculate_mx_elements:
            
            mx_elements = {} # Initialize the dictionary.

            for operator_name in self.operator_names:

                mx_elements[operator_name] = [[ np.vdot(ekets_i.ravel(), system_operators[operator_name].dot(ekets_j.ravel())) 
                                                    for ekets_i in ekets] 
                                                    for ekets_j in ekets]

            diag_result.update(mx_elements) # update the list.

            #Calculate coupling rates and dispersive shifts.
            
            cqed_elements = {}

            cqed_elements['g'] = [[ np.vdot(ekets_i.ravel(), system_operators['g_matrix'].dot(ekets_j.ravel())) 
                                                for ekets_i in ekets] 
                                                for ekets_j in ekets]


            chi_matrix = np.zeros(shape = (self.keig, self.keig), dtype='float64')
    
            for i in range(self.keig):
                for j in range(self.keig):
                    chi_matrix[i,j] = np.abs(cqed_elements['g'][i][j]) ** 2 / (evals[i] - evals[j] - self.f_resonator)
            
            cqed_elements['chi']   = np.sum(chi_matrix, 1) - np.sum(chi_matrix, 0) # dispersive shifts
            cqed_elements['kappa'] = np.sum(chi_matrix, 1) # Lamb shift

            diag_result.update(cqed_elements) # update the list.

        return diag_result



    def sweep_diagonalize(self, sweep_variable, calculate_mx_elements=False, progressbar_disable=False):
        """Diagonalizes the Hamiltonian and updates the attributes of the qubit at the parameters corresponding to the sweep variable.

        Args:
            sweep_variable (str): Name of the parameter to sweep.
            calculate_mx_elements (bool): Calculate the matrix elements, dispersive shifts, etc. if True.
            rogressbar_disable (bool): Display progress bar if it is False.

        Returns:
           None. 

        """

        # Define the sweep vector. 
        sweep_vector = getattr(self, sweep_variable + '_ls') 
        
        sweep_results = []
        for sweep_point in tqdm(sweep_vector, disable=progressbar_disable):

            self.__dict__[sweep_variable] = sweep_point # Update the attribute.

            # Calculate evals, ekets by diagonalizing the Hamiltonian.
            diag_result   = self.diagonalize(calculate_mx_elements) 

            sweep_results.append(diag_result) # Append the resultant dictionary to the sweep list.

        # Transform the list to attributes.
        for key in sweep_results[0].keys():   
            setattr(self, key, self.extract_results(key, sweep_results))

        return



    def sweep_diagonalize_full_Hamiltonian(self, sweep_variable):
        """Diagonalizes the coupled circuit and resonator Hamiltonian.

        Args:
            sweep_variable (str): Name of the parameter to sweep.

        Returns:
           None. 

        """

        a = tensor(destroy(sim_settings.NFOCK), qeye(self.keig)) # annihilation operator
        adg = a.dag() # creation operator

        # Create the resonator Hamiltonian in the resonator + qubit basis. 
        H_resonator = self.f_resonator * (adg * a)

        sweep_vector = getattr(self, sweep_variable + '_ls')
        
        sweep_evals = []
        sweep_idx = 0
        for sweep_point in tqdm(sweep_vector):

            # Create the qubit Hamiltonian in the resonator + qubit basis.
            H = Qobj(diags(self.evals[sweep_idx,:]))
            H_qubit = tensor(qeye(sim_settings.NFOCK), H)
            
            # Create the coupling Hamiltonian in the resonator + qubit basis.
            H_coupling = tensor(qeye(sim_settings.NFOCK), Qobj(self.g[sweep_idx,:,:])) * (a + adg) 

            # Create the full Hamiltonian.
            H_full = H_qubit + H_resonator + H_coupling

            # Diagonalize the full Hamiltonian.
            evals, _ = H_full.eigenstates(sparse=False, sort='low', eigvals=self.keig*sim_settings.NFOCK, tol=sim_settings.DIAG_TOL)
            evals    = evals - evals[0] 

            sweep_evals.append(dict(evals_full=evals)) # Append the result to the list.
            sweep_idx += 1

        # Convert the list to the attribute.
        setattr(self, 'evals_full', self.extract_results('evals_full', sweep_evals))

        return


    def extract_results(self, key, results):
        """Converts the list obtained from the parameter sweep to more convenient format.

        Args:
            key (str): Name of the key.
            results (list): Result of the parameter sweep.


        Returns:
            ndarray: the items converted into ndarray form. 

        """

        if key in ['evals', 'evals_full', 'ekets']:

            return np.array([[results[sweep_idx][key][level_idx] 
                            for level_idx in range(self.keig)] 
                            for sweep_idx in range(len(results))])

        else:
            return np.array([results[sweep_idx][key]
                            for sweep_idx in range(len(results))])


