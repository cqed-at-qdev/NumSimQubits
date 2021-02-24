# gaussian_pulse.py
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

class Gaussian_pulse:
    """ Class representing a gaussian pulse.

    Args:
        t_0 (float): The center of the pulse [s].
        sigma (float): The width of the pulse. [s] 
        tcutoff (float): The cutoff the pulse in the units of sigma.
        amp (float): The amplitude of the pulse [GHz].
        frequency (float): The frequency of the pulse [GHz]. 
        phase (float): The phase of the pulse. 
        t_ls (ndarray): The timeseries where the pulse is defined. 

    Attributes:
        t_0 (float): The center of the pulse [s].
        sigma (float): The width of the pulse. [s] 
        tcutoff (float): The cutoff the pulse in the units of sigma.
        amp (float): The amplitude of the pulse [GHz].
        frequency (float): The frequency of the pulse [GHz]. 
        phase (float): The phase of the pulse.
        t_ls (ndarray): The timeseries where the pulse is defined. 
        I (Cubic_Spline): The pulse shape defined in the I quadrature.
        Q (Cubic_Spline): The pulse shape defined in the Q quadrature.

    Methods:
        update_base(): Updates the shape of the pulse in I-Q plane.
      
    """

    def __init__(self, t_0, sigma, amp, freq, phase, t_ls, t_cutoff=4.0):
        """
        Gaussianpulse constructor.
        """

        self.t_0       = t_0
        self.sigma     = sigma
        self.amp       = amp
        self.freq      = freq 
        self.phase     = phase
        self.t_cutoff  = t_cutoff
        self.t_ls      = t_ls

        # Define the shape of the pulse in I-Q plane.
        self.update_base()

    def __repr__(self):
        """
        Gives complete information on the Gaussianpulse object.
        """
        return 'Gaussianpulse(\n t_0=%r, \n sigma=%r, \n amp=%r, \n freq=%r, \n phase=%r, \n t_cutoff=%r \n)' % (
            self.t_0, self.sigma, self.amp, self.freq, self.phase, self.t_cutoff)

    def update_base(self):
        '''Updates the shape of the pulse in I-Q plane.'''

        t_cutoff = self.t_cutoff * self.sigma

        offset = np.exp(- (t_cutoff / 2) ** 2 / (2 * self.sigma ** 2))
        norm = 1 - offset
    
        gaussian_shape =  2 * np.pi * 1e9 * self.amp * np.exp(1j * self.phase) * (np.exp(- (self.t_ls - self.t_0) ** 2 / (2 * self.sigma ** 2)) - offset) / norm
        gaussian_pulse =  gaussian_shape * ((self.t_ls > self.t_0 - t_cutoff / 2) & (self.t_ls < self.t_0 + t_cutoff / 2))
    
        self.I = Cubic_Spline(self.t_ls[0], self.t_ls[-1], np.real(gaussian_pulse))
        self.Q = Cubic_Spline(self.t_ls[0], self.t_ls[-1], np.imag(gaussian_pulse))
    
        return 
