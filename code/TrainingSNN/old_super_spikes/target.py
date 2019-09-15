# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:06:53 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

nt = 1800
N_out = 5
y = np.zeros((nt, N_out))
y[500, 0] = 1

target = []
target.append([50,250,350,400,450,500,550,650,850,950,1000,1050,1100,1300,1350,1400,1550,1750])
target.append([50,100,250,350,650,850,950,1150,1250,1450,1550,1600,1750])
target.append([50,150,250,350,400,450,500,550,650,850,950,1000,1050,1100,1250,1450,1550,1650,1750])
target.append([50,200,250,350,650,850,950,1100,1250,1450,1550,1700,1750])
target.append([50,250,350,400,450,500,550,700,750,800,950,1150,1300,1350,1400,1550,1750])

for i in range(N_out):    
    y[target[int(N_out-i-1)], i] = 1

t = np.arange(nt)
plt.figure(figsize=(8, 2))
for j in range(N_out):
    plt.scatter(t, y[:, j]*(j+1), marker="o", color="k")
plt.title('LIF neurons')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index') 
plt.ylim(0.5, N_out+0.5)
plt.tight_layout()
plt.show()
