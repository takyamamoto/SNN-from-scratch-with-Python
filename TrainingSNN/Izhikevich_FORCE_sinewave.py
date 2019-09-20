# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from Models.Neurons import IzhikevichNeuron
from Models.Synapses import DoubleExponentialSynapse

np.random.seed(seed=0)

#################
## モデルの定義  ###
#################
N = 2000  # ニューロンの数
dt = 0.04 # タイムステップ

C = 250      # 膜容量(pF)
a = 0.01     # 回復時定数の逆数 (1/ms)
b = -2       # uのvに対する共鳴度合い(pA/mV)
k = 2.5      # ゲイン (pA/mV)
d = 200      # 発火で活性化される正味の外向き電流(pA)
vrest = -60  # 静止膜電位 (mV) 
vreset = -65 # リセット電位 (mV) 
vthr = -20   # 閾値電位 (mV)
vpeak = 30   #　ピーク電位 (mV)
td = 20; tr = 2 # シナプスの時定数
P = np.eye(N)*2 # 相関行列の逆行列の初期化

# 教師信号(正弦波)の生成
T = 15000 # シミュレーション時間 (s)
tmin = round(5000/dt) # 重み更新の開始ステップ
tcrit = round(10000/dt) # 重み更新の終了ステップ
step = 50 # 重み更新のステップ間隔
nt = round(T/dt) # シミュレーションステップ数
Q = 5e3; G = 5e3

zx = np.sin(2*math.pi*np.arange(nt)*dt*5*1e-3) # 教師信号

# ニューロンとシナプスの定義 
neurons = IzhikevichNeuron(N=N, dt=dt, C=C, a=a, b=b,
                           k=k, d=d, vrest=vrest, vreset=vreset,
                           vthr=vthr, vpeak=vpeak)
neurons.v = vrest + np.random.rand(N)*(vpeak-vrest) # 膜電位の初期化

synapses_out = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)
synapses_rec = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)

# 再帰重みの初期値
p = 0.1 # ネットワークのスパース性
OMEGA = G*(np.random.randn(N,N))*(np.random.rand(N,N)<p)/(math.sqrt(N)*p)

for i in range(N):
    QS = np.where(np.abs(OMEGA[i,:])>0)[0]
    OMEGA[i,QS] = OMEGA[i,QS] - np.sum(OMEGA[i,QS], axis=0)/len(QS)


# 変数の初期値
k = 1 # 出力ニューロンの数
E = (2*np.random.rand(N,k) - 1)*Q
PSC = np.zeros(N) # シナプス後電流
JD = np.zeros(N) # 再帰入力の重み和
z = np.zeros(k) # 出力の初期化
Phi = np.zeros(N) #　学習される重みの初期値

# 記録用変数 
REC_v = np.zeros((nt,10)) # 膜電位の記録変数
current = np.zeros(nt) # 出力の電流の記録変数
tspike = np.zeros((5*nt,2)) # スパイク時刻の記録変数
ns = 0 # スパイク数の記録変数 

BIAS = 1000 # 入力電流のバイアス

#################
## シミュレーション ###
#################
for t in tqdm(range(nt)):
    I = PSC + np.dot(E, z) + BIAS # シナプス電流 
    s = neurons(I) # 中間ニューロンのスパイク
    
    index = np.where(s)[0] # 発火したニューロンのindex
    
    len_idx = len(index)
    if len_idx > 0:
        JD = np.sum(OMEGA[:, index], axis=1)  
        tspike[ns:ns+len_idx,:] = np.vstack((index, 0*index+dt*t)).T
        ns = ns + len_idx # スパイク数の記録

    PSC = synapses_rec(JD*(len_idx>0)) # 再帰的入力電流
    #PSC = OMEGA @ r # 遅い

    r = synapses_out(s) # 出力電流(神経伝達物質の放出量)  
    r = np.expand_dims(r,1) # (N,) -> (N, 1)
    
    z = Phi.T @ r # デコードされた出力
    err = z - zx[t] # 誤差

    # FORCE法(RLS)による重み更新
    if t % step == 1:
        if t > tmin:
            if t < tcrit:
                cd = (P @ r)
                Phi = Phi - (cd @ err.T)
                P = P - (cd @ cd.T) / (1.0 + r.T @ cd)
    
    current[t] = z
    REC_v[t] = neurons.v_[:10]

#################
#### 結果表示 ####
#################
TotNumSpikes = ns 
M = tspike[tspike[:,1]>dt*tcrit,:]
AverageRate = len(M)/(N*(T-dt*tcrit))*1e3
print("\n")
print("Total number of spikes : ", TotNumSpikes)
print("Average firing rate(Hz): ", AverageRate)

def hide_ticks(): #上と右の軸を表示しないための関数
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

step_range = 20000
plt.figure(figsize=(10, 5))
plt.subplot(1,2,1)
for j in range(5):
    plt.plot(np.arange(step_range)*dt*1e-3,
             REC_v[:step_range, j]/(50-vreset)+j,
             color="k")
hide_ticks()
plt.title('Pre-Learning')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index') 
plt.tight_layout()

plt.subplot(1,2,2)
for j in range(5):
    plt.plot(np.arange(nt-step_range, nt)*dt*1e-3,
             REC_v[nt-step_range:, j]/(50-vreset)+j,
             color="k")
hide_ticks()
plt.title('Post Learning')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig("Iz_FORCE_prepost.pdf")
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(nt)*dt*1e-3, zx,
         label="Target", color="k")
plt.plot(np.arange(nt)*dt*1e-3, current,
         label="Decoded output",
         linestyle="dashed", color="k")
plt.xlim(4.5,5.5)
plt.ylim(-1.1,1.4)
hide_ticks()
plt.title('Pre/peri Learning')
plt.xlabel('Time (s)')
plt.ylabel('current') 
plt.tight_layout()

plt.subplot(1,2,2)
plt.title('Post Learning')
plt.plot(np.arange(nt)*dt*1e-3, zx,
         label="Target", color="k")
plt.plot(np.arange(nt)*dt*1e-3, current,
         label="Decoded output",
         linestyle="dashed", color="k")
plt.xlim(14,15)
plt.ylim(-1.1,1.4)
hide_ticks()
plt.xlabel('Time (s)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("Iz_FORCE_decoded.pdf")
plt.show()

"""
Z = np.linalg.eig(OMEGA + np.expand_dims(E,1) @ np.expand_dims(Phi,1).T)
Z2 = np.linalg.eig(OMEGA)
plt.figure(figsize=(6, 5))
plt.title('Weight eigenvalues')
plt.scatter(Z2[0].real, Z2[0].imag, c='r', s=5, label='Pre-Learning')
plt.scatter(Z[0].real, Z[0].imag, c='k', s=5, label='Post-Learning')
plt.legend()
plt.xlabel('Real')
plt.ylabel('Imaginary')
#plt.savefig("LIF_weight_eigenvalues.png")
plt.show()
"""