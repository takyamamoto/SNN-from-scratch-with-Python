# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Models.Neurons import IzhikevichNeuron

dt = 0.5 # ms
T = 1000 # ms
nt = round(T/dt)

C = 100
a = 0.03
b = -2
k = 0.7
d = 100
vrest = -60
vreset = -50
vthr = -40
vpeak = 35

BIAS = 0
I = 70
neuron = IzhikevichNeuron(N=1, dt=dt, C=C, a=a, b=b,
                          k=k, d=d, vrest=vrest, vreset=vreset,
                          vthr=vthr, vpeak=vpeak)
v_list = []
for i in tqdm(range(nt)):
    s = neuron(I)
    v_list.append(neuron.v[0])

plt.plot(np.array(v_list))
