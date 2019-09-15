# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)

dt = 1e-3; T = 1; nt = round(T/dt) # シミュレーション時間
n_neurons = 10 # ニューロンの数

fr = 30 # ポアソンスパイクの発火率(Hz)
isi = np.random.exponential(1/(fr*dt),
                            size=(round(nt*1.5/fr), n_neurons))
spike_time = np.cumsum(isi, axis=0) # ISIを累積
spike_time[spike_time > nt - 1] = 0 # ntを超える場合を0に
spike_time = spike_time.astype(np.int32) # float to int
spikes = np.zeros((nt, n_neurons)) # スパイク記録変数
for i in range(n_neurons):    
    spikes[spike_time[:, i], i] = 1
spikes[0] = 0 # (spike_time=0)の発火を削除
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
plt.savefig("poisson_process.pdf", dpi=300)
plt.show()
