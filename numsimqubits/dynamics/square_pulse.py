# square_pulse.py
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

class Square_pulse:
    """ Class representing a square pulse.

    Args:
        t_0 (float): The center of the pulse [s].
        T (float): The width of the pulse. [s] 
        amp (float): The amplitude of the pulse [GHz].
        frequency (float): The frequency of the pulse [GHz]. 
        phase (float): The phase of the pulse.
        t_ls (ndarray): The timeseries where the pulse is defined.  

    Attributes:
        t_0 (float): The center of the pulse [s].
        T (float): The width of the pulse. [s] 
        amp (float): The amplitude of the pulse [GHz].
        frequency (float): The frequency of the pulse [GHz]. 
        phase (float): The phase of the pulse.
        t_ls (ndarray): The timeseries where the pulse is defined. 
        I (Cubic_Spline): The pulse shape defined in the I quadrature.
        Q (Cubic_Spline): The pulse shape defined in the Q quadrature.

    Methods:
        update_base(): Updates the shape of the pulse in I-Q plane.  

    """

    def __init__(self, t_0, T, amp, freq, phase, t_ls):
        """
        Gaussianpulse constructor.
        """

        self.t_0       = t_0
        self.T         = T
        self.amp       = amp
        self.freq      = freq 
        self.phase     = phase
        self.t_ls      = t_ls
        
        # Define the shape of the pulse in I-Q plane.
        self.update_base()
        
    def __repr__(self):
        """
        Gives complete information on the Gaussianpulse object.
        """
        return 'Gaussianpulse(\n t_0=%r, \n T=%r, \n amp=%r, \n freq=%r, \n phase=%r, \n)' % (
            self.t_0, self.T, self.amp, self.freq, self.phase)

    def update_base(self):
        '''Updates the shape of the pulse in I-Q plane.'''

        square_shape = 2 * np.pi * 1e9 * self.amp * np.exp(1j * self.phase)
        square_pulse = square_shape * ((self.t_ls > self.t_0 - self.T / 2) & (self.t_ls < self.t_0 + self.T / 2))
    
        self.I = Cubic_Spline(self.t_ls[0], self.t_ls[-1], np.real(square_pulse))
        self.Q = Cubic_Spline(self.t_ls[0], self.t_ls[-1], np.imag(square_pulse))

        return
    