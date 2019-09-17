# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Models.Neurons import CurrentBasedLIF
from Models.Synapses import DoubleExponentialSynapse

np.random.seed(seed=0)

#################
## モデルの定義  ###
#################
N = 2000  # ニューロンの数
dt = 5e-5 # タイムステップ(s)
tref = 2e-3 # 不応期(s)
tc_m = 1e-2 #　膜時定数(s)
vreset = -65 # リセット電位(mV) 
vrest = 0 # 静止膜電位(mV)
vthr = -40 # 閾値電位(mV)
vpeak = 30 # ピーク電位(mV)
BIAS = -40 # 入力電流のバイアス(pA)
td = 2e-2; tr = 2e-3 # シナプスの時定数(s)
alpha = dt*0.1  
P = np.eye(N)*alpha
Q = 10; G = 0.04

# 教師信号(正弦波)の生成
T = 15 # シミュレーション時間 (s)
tmin = round(5/dt) # 重み更新の開始ステップ
tcrit = round(10/dt) # 重み更新の終了ステップ
step = 50 # 重み更新のステップ間隔
nt = round(T/dt) # シミュレーションステップ数
zx = np.sin(2*np.pi*np.arange(nt)*dt*5) # 教師信号

# ニューロンとシナプスの定義 
neurons = CurrentBasedLIF(N=N, dt=dt, tref=tref, tc_m=tc_m,
                          vrest=vrest, vreset=vreset, vthr=vthr, vpeak=vpeak)
neurons.v = vreset + np.random.rand(N)*(vpeak-vreset) # 膜電位の初期化

synapses_out = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)
synapses_rec = DoubleExponentialSynapse(N, dt=dt, td=td, tr=tr)

# 再帰重みの初期値
p = 0.1 # ネットワークのスパース性
OMEGA = G*(np.random.randn(N,N))*(np.random.rand(N,N)<p)/(np.sqrt(N)*p)
for i in range(N):
    QS = np.where(np.abs(OMEGA[i,:])>0)[0]
    OMEGA[i,QS] = OMEGA[i,QS] - np.sum(OMEGA[i,QS], axis=0)/len(QS)


# 変数の初期値
k = 1 # 出力ニューロンの数
E = (2*np.random.rand(N, k) - 1)*Q
PSC = np.zeros(N).astype(np.float32) # シナプス後電流
JD = np.zeros(N).astype(np.float32) # 再帰入力の重み和
z = np.zeros(k).astype(np.float32) # 出力の初期化
Phi = np.zeros(N).astype(np.float32) #　学習される重みの初期値

# 記録用変数 
REC_v = np.zeros((nt,10)).astype(np.float32) # 膜電位の記録変数
current = np.zeros(nt).astype(np.float32) # 出力の電流の記録変数
tspike = np.zeros((4*nt,2)).astype(np.float32) # スパイク時刻の記録変数
ns = 0 # スパイク数の記録変数 

#################
## シミュレーション ###
#################
for t in tqdm(range(nt)):
    I = PSC + np.dot(E, z) + BIAS # シナプス電流 
    s = neurons(I) # 中間ニューロンのスパイク
    
    index = np.where(s)[0] # 発火したニューロンのindex
    
    len_idx = len(index) # 発火したニューロンの数
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
    
    current[t] = z # デコード結果の記録
    REC_v[t] = neurons.v_[:10] # 膜電位の記録

#################
#### 結果表示 ####
#################
TotNumSpikes = ns 
M = tspike[tspike[:,1]>dt*tcrit,:]
AverageRate = len(M)/(N*(T-dt*tcrit))
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
    plt.plot(np.arange(step_range)*dt,
             REC_v[:step_range, j]/(50-vreset)+j,
             color="k")
hide_ticks()
plt.title('Pre-Learning')
plt.xlabel('Time (s)')
plt.ylabel('Neuron Index') 
plt.tight_layout()

plt.subplot(1,2,2)
for j in range(5):
    plt.plot(np.arange(nt-step_range, nt)*dt,
             REC_v[nt-step_range:, j]/(50-vreset)+j,
             color="k")
hide_ticks()
plt.title('Post Learning')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.savefig("LIF_FORCE_prepost.pdf")
plt.show()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(np.arange(nt)*dt, zx,
         label="Target", color="k")
plt.plot(np.arange(nt)*dt, current,
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
plt.plot(np.arange(nt)*dt, zx,
         label="Target", color="k")
plt.plot(np.arange(nt)*dt, current,
         label="Decoded output",
         linestyle="dashed", color="k")
plt.xlim(14,15)
plt.ylim(-1.1,1.4)
hide_ticks()
plt.xlabel('Time (s)')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("LIF_FORCE_decoded.pdf")
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