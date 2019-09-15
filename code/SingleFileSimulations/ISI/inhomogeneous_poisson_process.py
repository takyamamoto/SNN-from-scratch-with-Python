# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)

dt = 1e-3; T = 1; nt = round(T/dt) # シミュレーション時間
n_neuron = 10 # ニューロンの数
t = np.arange(nt)*dt

# ポアソンスパイクの発火率(Hz)
fr = np.expand_dims(30*np.sin(10*t)**2, 1)

# スパイク記録変数
spikes = np.where(np.random.rand(nt, n_neuron) < fr*dt, 1, 0)

print("Num. of spikes:", np.sum(spikes))
# 描画
plt.figure(figsize=(5, 4))
plt.subplot(2,1,1)
plt.plot(t, fr[:, 0], color="k")
plt.ylabel('Firing rate (Hz)') 
plt.xlim(0, T)

plt.subplot(2,1,2)
for i in range(n_neuron):    
    plt.plot(t, spikes[:, i]*(i+1), 'ko', markersize=2,
             rasterized=True)
plt.xlabel('Time (s)')
plt.ylabel('Neuron index') 
plt.xlim(0, T)
plt.ylim(0.5, n_neuron+0.5)
plt.tight_layout()
plt.savefig("inhomogenous_poisson_process.pdf", dpi=300)
plt.show()

