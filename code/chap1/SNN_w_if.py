# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:19:15 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

def step(x):
    return 1 * (x > 0)

num_in = 10
num_out = 2

W = np.random.rand(num_out, num_in)

fr = 60 # Hz

T = 100 # simulation time step
V_th = 2.5
tau = 25e-3 # sec
dt = 1e-3 # sec
l = (1-dt/tau)

x = np.where(np.random.rand(T, num_in) < fr*dt, 1, 0)
y = np.zeros((T, num_out))
s = np.zeros((T, num_out))

for t in range(T-1):
    s[t+1] = relu(W@x[t] + l*s[t]*(1-y[t]))
    y[t+1] = step(s[t+1]-V_th)

plt.plot(s[:,0])
plt.plot(s[:,1])
plt.show()