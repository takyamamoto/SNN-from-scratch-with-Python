# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


class LIF(object):
    def __init__(self, num_in=10, num_out=1, fr_in=60, T=100,
                 V_th=2.5, tau=2.5e-2, dt=1e-3):
        self.num_in = num_in
        self.num_out = num_out 
        self.fr_in = fr_in # input poission spike firing rate (Hz)
        self.T = T # simulation time step
        self.V_th = V_th # threshold
        self.tau = tau # membrane time constant(sec)
        self.dt = dt # sec
    
    def step(self, x):
        return 1 * (x > 0)
    
    def initialize_weights(self):
        self.W = np.random.rand(self.num_out, self.num_in)

    def initialize_states(self):
        self.v = np.zeros((self.num_out))
        self.y = np.zeros((self.num_out))
        
    def forward(self):
        self.initialize_weights()
        self.initialize_states()
        
        x = np.where(np.random.rand(self.T, self.num_in) < self.fr_in * self.dt, 1, 0)
        
        for t in range(self.T-1):
            v = self.W @ x[t] + (1 - self.dt/self.tau)*self.v*(1 - self.y)
            y = self.step(v - self.V_th)
            
            self.v = v
            self.y = y
        
        return y
        #plt.plot(v[0])
        #plt.show()

if __name__ == '__main__':
    model = LIF()
    model.forward()