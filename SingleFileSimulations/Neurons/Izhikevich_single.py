# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

dt = 0.5; T = 400 # ms
nt = round(T/dt) # シミュレーションのステップ数

# Regular spiking (RS) neurons
C = 100      # 膜容量(pF)
a = 0.03     # 回復時定数の逆数 (1/ms)
b = -2       # uのvに対する共鳴度合い(pA/mV)
k = 0.7      # ゲイン (pA/mV)
d = 100      # 発火で活性化される正味の外向き電流(pA)
vrest = -60  # 静止膜電位 (mV) 
vreset = -50 # リセット電位 (mV) 
vthr = -40   # 閾値電位 (mV)
vpeak = 35   #　ピーク電位 (mV)
t = np.arange(nt)*dt
I = 100*(t>50) - 100*(t>350) # 入力電流(pA)

"""
# Intrinsically Bursting (IB) neurons
C = 150; a = 0.01; b = 5; k =1.2; d = 130
vrest = -75; vreset = -56; vthr = -45; vpeak = 50;
I = 600*(t>50) - 600*(t>350) 

# Chattering (CH) or fast rhythmic bursting (FRB) neurons
C = 50; a = 0.03; b = 1; k =1.5; d = 150
vrest = -60; vreset = -40; vthr = -40; vpeak = 35;
I = 600*(t>50) - 600*(t>350) 
"""

# 初期化(膜電位, 膜電位(t-1), 回復電流)
v = vrest; v_ = v; u = 0
v_arr = np.zeros(nt) # 膜電位を記録する配列
u_arr = np.zeros(nt) # 回復変数を記録する配列

# シミュレーション
for i in tqdm(range(nt)):
    dv = (k*(v - vrest)*(v - vthr) - u + I[i]) / C
    v = v + dt*dv # 膜電位の更新
    u = u + dt*(a*(b*(v_-vrest)-u)) # 膜電位の更新
        
    s = 1*(v>=vpeak) #発火時は1, その他は0の出力
        
    u = u + d*s # 発火時に回復変数を上昇
    v = v*(1-s) + vreset*s # 発火時に膜電位をリセット
    v_ = v # v(t-1) <- v(t)

    v_arr[i] = v  # 膜電位の値を保存
    u_arr[i] = u  # 回復変数の値を保存
    
# 描画
plt.figure(figsize=(5, 5))
plt.subplot(2,1,1)
plt.plot(t, v_arr, color="k")
#plt.title("Regular spiking (RS) neurons")
plt.ylabel('Membrane potential (mV)') 
plt.xlim(0, T)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(t, u_arr, color="k")
plt.xlabel('Time (ms)')
plt.ylabel('Recovery current (pA)')
plt.xlim(0, T) 
plt.tight_layout()
plt.savefig('Izhikevich_regular.pdf')
plt.show()