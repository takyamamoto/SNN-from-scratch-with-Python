# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
 
dt = 5e-5 # 時間ステップ (s)
T = 1 # シミュレーション時間 (s)
nt = round(T/dt) #Time steps
 
tref = 5e-3 # 不応期 (s)
tc_m = 1e-2 # 膜時定数 (s)
vrest = 0 # 静止膜電位 (mV) 
vreset = 0 # リセット電位 (mV) 
vthr = 1 # 閾値電位 (mV)

I_max = 3 # (nA)
N = 100

# 入力電流
I = np.linspace(0, I_max, N) # pA
spikes = np.zeros((N, nt))

for i in tqdm(range(N)):
    # 初期化
    v = vreset
    tlast = 0

    # シミュレーション
    for t in range(nt):
        # 更新
        dv = (vrest - v + I[i]) / tc_m
        update = 1 if ((dt*t) > (tlast + tref)) else 0
        v = v + update*dv*dt
        
        # 発火の確認    
        s = 1 if (v>=vthr) else 0 #発火時は1, その他は0の出力
        tlast = tlast*(1-s) + dt*t*s
         
        # 保存
        spikes[i, t] = s 
         
        # リセット
        v = v*(1-s) + vreset*s

# 描画
rate = np.sum(spikes, axis=1) / T
plt.figure(figsize=(4, 3))
plt.plot(I, rate, color="k")
plt.xlabel('Input current (nA)')
plt.ylabel('Firing rate (Hz)') 
plt.xlim(0, I_max)
plt.tight_layout()
plt.savefig('LIF_FI_numerical.pdf')
#plt.show()