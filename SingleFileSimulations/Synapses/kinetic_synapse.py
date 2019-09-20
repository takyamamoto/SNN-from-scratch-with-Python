# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
 
dt = 1e-4 # 時間ステップ (sec)
alpha = 1/5e-4 # synaptic decay time (sec)
beta = 1/5e-3 # synaptic rise time (sec)
T = 0.05 # シミュレーション時間 (sec)
nt = round(T/dt) #  シミュレーションの総ステップ
 
r = 0 # 初期値
single_r = [] #記録用配列 
for t in range(nt):    
    spike = 1 if t == 0 else 0
    single_r.append(r)
    r += (alpha*spike*(1-r) - beta*r)*dt

# 描画
time = np.arange(nt)*dt
plt.figure(figsize=(4, 3))
plt.plot(time, np.array(single_r), color="k")
plt.xlabel('Time (s)'); plt.ylabel('Post-synaptic current (pA)') 
plt.tight_layout()
plt.savefig('kinetic_synapse.pdf')
plt.show()