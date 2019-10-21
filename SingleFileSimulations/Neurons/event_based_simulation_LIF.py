# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 00:17:20 2019

@author: user
"""
# http://www.scholarpedia.org/article/Chaos_in_neurons
# Poincare sections of chaotic networks 

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def h(phase):
    return np.sign(phase)*np.exp(np.abs(phase) - 1)

def inv_h(x):
    return np.sign(x)*np.log(1+np.abs(x))

# number of spikes in calculation
nspikes = int(1e5)

# define adjacency matrix
topo = np.array([[0,0,0],
                 [1,0,1],
                 [0,1,0]])

# define effective coupling strength
c = 1

#initial network state
phi = np.random.rand(3)   

# initialize phase history
phi_arr = np.zeros((nspikes, 3))

for t in tqdm(range(0, nspikes)):
    # find next spiking neuron j
    phi_max = np.max(phi)
    j = np.argmax(phi)
    
    # calculate next spike time
    dt = np.pi*0.5 - phi_max
    
    # evolve phases till next spike time
    phi += dt
    
    # get postsynaptic neurons
    post_idx = np.where(topo[:, j] > 0)[0]
    
    # update postsynaptic neurons (leaky integrate and fire neuron)
    phi[post_idx] = inv_h(h(phi[post_idx]) - c) 
    # reset spiking neuron
    phi[j] = -np.pi*0.5 
    phi_arr[t] = phi

idx = np.where(phi_arr[:, 0] == -np.pi*0.5)
plt.figure(figsize=(5, 4))
plt.scatter(phi_arr[idx, 1], phi_arr[idx, 2], s=1,
            alpha=1, color="k")
plt.axis('off')
plt.tight_layout()
plt.show()