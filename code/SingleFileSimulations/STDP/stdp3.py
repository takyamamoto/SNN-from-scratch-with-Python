# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 13:52:03 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(seed=0)

dt = 1e-3; T = 5e-2 #sec
nt = round(T/dt)

N_pre = nt; N_post = 2

tau_p = tau_m = 2e-2 #ms
A_p = 0.01; A_m = 1.05*A_p

# pre/postsynaptic spikes
spike_pre = np.eye(N_pre) #単位行列でdtごとに発火するニューロンをN個作成
spike_post = np.zeros((N_post, nt))
spike_post[0, -1] = spike_post[1, 0] = 1

# 初期化
x_pre = np.zeros(N_pre)
x_post = np.zeros(N_post)
W = np.zeros((N_post, N_pre))

for t in range(nt):
    # 1次元配列 -> 縦ベクトル or 横ベクトル
    spike_pre_ = np.expand_dims(spike_pre[:, t], 0) # (1, N)
    spike_post_ = np.expand_dims(spike_post[:, t], 1) # (2, 1)
    x_pre_ = np.expand_dims(x_pre, 0) # (1, N)
    x_post_ = np.expand_dims(x_post, 1) # (2, 1)
    
    # Online STDP
    dW = A_p*np.matmul(spike_post_, x_pre_)
    dW -= A_m*np.matmul(x_post_, spike_pre_)
    W += dW
    
    # Update
    x_pre = x_pre*(1-dt/tau_p) + spike_pre[:, t]
    x_post = x_post*(1-dt/tau_m) + spike_post[:, t]


# 結果
delta_w = np.zeros(nt*2-1) # スパイク時間差 = 0msが重複
delta_w[:nt] = W[0, :]; delta_w[nt:] = W[1, 1:]

# 描画
time = np.arange(-T, T-dt, dt)*1e3
plt.figure(figsize=(5, 4))
plt.plot(time, delta_w[::-1])
plt.hlines(0, -50, 50)
plt.xlabel("$\Delta t$ (ms)")
plt.ylabel("$\Delta w$")
plt.xlim(-50, 50)
plt.tight_layout()
plt.savefig('online_stdp2.pdf')
#plt.show()
