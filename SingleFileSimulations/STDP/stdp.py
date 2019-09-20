# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:52:03 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

tau_p = tau_m = 20 #ms
A_p = 0.01; A_m = 1.05*A_p;
dt = np.arange(-50, 50, 1) #ms
dw = A_p*np.exp(-dt/tau_p)*(dt>0) - A_m*np.exp(dt/tau_p)*(dt<0) 

plt.figure(figsize=(5, 4))
plt.plot(dt, dw, color="k")
plt.hlines(0, -50, 50)
plt.xlabel("$\Delta t$ (ms)")
plt.ylabel("$\Delta w$")
plt.xlim(-50, 50)
plt.tight_layout()
plt.savefig('stdp.pdf')
#plt.show()