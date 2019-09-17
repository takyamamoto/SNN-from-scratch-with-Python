# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:43:20 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def NoisySoftplus(x, k=0.16, sigma=0.2):
    ks = k*sigma
    return ks*np.log(1+np.exp(x/ks))

x = np.arange(-0.6, 0.6, 0.01)
for i in range(5):
    y = NoisySoftplus(x, sigma=0.5*(i+1))
    plt.plot(x, y)
plt.show()

 