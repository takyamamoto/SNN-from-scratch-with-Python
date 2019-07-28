# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 11:41:18 2019

@author: yamamoto
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1
dt = 1e-3
T = 5
nt = round(T/dt)

ts = np.arange(0, T, dt)
#fr = 60

fr = 20*(1+np.sin(4.0 * np.pi*ts))
x = np.where(np.random.rand(N, nt) < fr*dt, 1, 0)

plt.plot(ts, x[0])
plt.plot(ts, fr/100)
"""
dt = 0.0001
L = 100000
ts = np.linspace(0, 10, L)
f = 1.0
lam = 20*(1+np.sin(2.0 * np.pi * f * ts))


def pprocess_inhomopoisson(lam):
    x = []; # push!(x, num)
    rs = np.zeros(L)

    R = 0;

    rs[1] = lam[1];

    eta = - np.log(np.random.rand());
    for i in range(2,L):
        r = lam[i]; # Intensity
        R = R + r*dt

        if eta <= R:
            x.append(i*dt) # Spike time

            R = 0.0;
            eta = - np.log(np.random.rand())

        rs[i] = r

    return x, rs

x, rs = pprocess_inhomopoisson(lam)
"""