# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Models.Neurons import CurrentBasedLIF
from Models.Connections import DelayConnection

dt = 1e-4; T = 5e-2
nt = round(T/dt)

neuron1 = CurrentBasedLIF(N=1, dt=dt, tc_m=1e-2, tref=0, 
                          vrest=0, vreset=0, vthr=1, vpeak=1)
neuron2 = CurrentBasedLIF(N=1, dt=dt, tc_m=1e-1, tref=0,
                          vrest=0, vreset=0, vthr=1, vpeak=1)
delay_connect = DelayConnection(N=1, delay=2e-3, dt=dt)

I = 2
v_arr1 = np.zeros(nt)
v_arr2 = np.zeros(nt)

# シミュレーション
for t in tqdm(range(nt)):
    # 更新
    s1 = neuron1(I)
    d1 = delay_connect(s1)
    s2 = neuron2(0.02/dt*d1)

    # 保存
    v_arr1[t] = neuron1.v_
    v_arr2[t] = neuron2.v_

time = np.arange(nt)*dt*1e3
plt.figure(figsize=(5, 4))
plt.plot(time, v_arr1, label="Neuron1", linestyle="dashed")
plt.plot(time, v_arr2, label="Neuron2")
plt.xlabel("Time (ms)"); plt.ylabel("v"); 
plt.legend(loc="upper left"); plt.tight_layout();
#plt.savefig('delay.pdf')
plt.show()

