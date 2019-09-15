# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
 
dt = 5e-5; T = 1; nt = round(T/dt)
tref = 5e-3; tc_m = 1e-2; vrest = 0; vreset = 0; vthr = 1

I_max = 3 # (nA)
N = 100 # N種類の入力電流
I = np.linspace(0, I_max, N) # 入力電流(pA)
spikes = np.zeros((N, nt)) # スパイクの記録変数

for i in tqdm(range(N)):
    v = vreset; tlast = 0 # 初期化
    for t in range(nt):
        dv = (vrest - v + I[i]) / tc_m #　膜電位の導関数
        update = 1 if ((dt*t) > (tlast + tref)) else 0 # 不応期でないかの確認
        v = v + update*dv*dt # 膜電位の更新
        s = 1 if (v>=vthr) else 0 #発火時は1, その他は0の出力
        tlast = tlast*(1-s) + dt*t*s # スパイク時刻の更新
        spikes[i, t] = s # 保存
        v = v*(1-s) + vreset*s # 膜電位のリセット

# 描画
rate = np.sum(spikes, axis=1) / T # 発火率
plt.figure(figsize=(5, 4))
plt.plot(I, rate, color="k")
plt.xlabel('Input current (nA)')
plt.ylabel('Firing rate (Hz)') 
plt.tight_layout()
plt.savefig('LIF_FI_numerical.pdf')
#plt.show()