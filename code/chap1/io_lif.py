# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 08:52:13 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

I = np.arange(0, 10, 0.1)
tau = 1e-2 #sec
R = 1
V_th = 1
delta_abs = 4e-3 #sec
rate = 1 / (delta_abs + tau*np.log(R*I/(R*I-V_th)))
rate[np.isnan(rate)] = 0
plt.plot(I, rate)