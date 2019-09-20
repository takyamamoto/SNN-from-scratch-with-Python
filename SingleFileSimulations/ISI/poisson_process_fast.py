# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)

dt = 1e-3; T = 1; nt = round(T/dt) # シミュレーション時間
n_neurons = 10 # ニューロンの数

fr = 30 # ポアソンスパイクの発火率(Hz)

# スパイク記録変数
spikes = np.where(np.random.rand(nt, n_neurons) < fr*dt, 1, 0)

print("Num. of spikes:", np.sum(spikes))
print("Firing rate:", np.sum(spikes)/(n_neurons*T))
# 描画
t = np.arange(nt)*dt
plt.figure(figsize=(5, 4))
for i in range(n_neurons):    
    plt.plot(t, spikes[:, i]*(i+1), 'ko', markersize=2,
             rasterized=True)
plt.xlabel('Time (s)')
plt.ylabel('Neuron index') 
plt.xlim(0, T)
plt.ylim(0.5, n_neurons+0.5)
plt.tight_layout()
plt.savefig("poisson_process_fast.pdf", dpi=300)

