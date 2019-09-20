# -*- coding: utf-8 -*-

import numpy as np

class SingleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=5e-3):
        """
        Args:
            td (float):Synaptic decay time
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.r = np.zeros(N)

    def initialize_states(self):
        self.r = np.zeros(self.N)

    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td) + spike/self.td
        self.r = r
        return r
     

class DoubleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
    
    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*(1-self.dt/self.td) + spike/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

