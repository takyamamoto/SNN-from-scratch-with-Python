# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def step(x):
    return 1 * (x > 0)

num_in = 10
num_out = 1

W = np.random.rand(num_out, num_in)

fr_in = 60 # input poission spike firing rate (Hz)
T = 100 # simulation time step
V_th = 2.5 # threshold
tau = 2.5e-2 # sec
dt = 1e-3 # sec

x = np.where(np.random.rand(T, num_in) < fr_in * dt, 1, 0)

v = np.zeros((T, num_out))
y = np.zeros((T, num_out))

for t in range(T-1):
    v[t+1] = W @ x[t] + (1 - dt/tau)*v[t]*(1 - y[t])
    y[t+1] = step(v[t+1] - V_th)

plt.plot(v[:,0])
plt.show()