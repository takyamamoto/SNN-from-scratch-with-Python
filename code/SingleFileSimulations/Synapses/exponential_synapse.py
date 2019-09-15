# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:33:12 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
 
dt = 5e-5 # 時間ステップ (sec)
td = 2e-2 # synaptic decay time (sec)
tr = 2e-3 # synaptic rise time (sec)
T = 0.1 # シミュレーション時間 (sec)
nt = round(T/dt) #  シミュレーションの総ステップ
 
# 単一指数関数型シナプス
r = 0 # 初期値
single_r = [] #記録用配列 
for t in range(nt):    
    if t == 0:
        spike = 1 # t=0のときにスパイク発生
    else:
        spike = 0
    single_r.append(r)
    r = r*(1-dt/td) + spike/td
    #r = r*np.exp(-dt/td) + spike/td
    
# 二重指数関数型シナプス
r = 0; hr = 0 # 初期値
double_r = [] #記録用配列
for t in range(nt):    
    if t == 0:
        spike = 1 # t=0のときにスパイク発生
    else:
        spike = 0
    double_r.append(r)
    r = r*(1-dt/tr) + hr*dt 
    hr = hr*(1-dt/td) + spike/(tr*td)
    #r = r*np.exp(-dt/tr) + hr*dt 
    #hr = hr*np.exp(-dt/td) + spike/(tr*td)
 
# 描画
time = np.arange(nt)*dt
plt.figure(figsize=(4, 3))
plt.plot(time, np.array(single_r), linestyle="dashed",
         label="single exponential")
plt.plot(time, np.array(double_r),  label="double exponential")
plt.xlabel('Time (s)'); plt.ylabel('Post-synaptic current (pA)') 
plt.legend()
plt.tight_layout()
#plt.savefig('exp_synapse.pdf')
plt.show()