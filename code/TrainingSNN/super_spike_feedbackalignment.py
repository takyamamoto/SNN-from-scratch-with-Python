# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Models.Neurons import CurrentBasedLIF
from Models.Synapses import DoubleExponentialSynapse
from Models.Connections import FullConnection, DelayConnection

np.random.seed(seed=0)
    
class ErrorSignal:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        self.dt = dt
        self.td = td
        self.tr = tr
        self.N = N
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
        self.b = (td/tr)**(td/(tr-td)) # 規格化定数
    
    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)    

    def __call__(self, output_spike, target_spike):
        r = self.r*(1-self.dt/self.tr) + self.hr/self.td*self.dt 
        hr = self.hr*(1-self.dt/self.td) + (target_spike - output_spike)/self.b
        
        self.r = r
        self.hr = hr
        
        return r

class EligibilityTrace:
    def __init__(self, N_in, N_out, dt=1e-4, td=1e-2, tr=5e-3):
        self.dt = dt
        self.td = td
        self.tr = tr
        self.N_in = N_in
        self.N_out = N_out
        self.r = np.zeros((N_out, N_in))
        self.hr = np.zeros((N_out, N_in))
    
    def initialize_states(self):
        self.r = np.zeros((self.N_out, self.N_in))
        self.hr = np.zeros((self.N_out, self.N_in))
    
    def surrogate_derivative_fastsigmoid(self, u, beta=1, vthr=-50):
        return 1 / (1 + np.abs(beta*(u-vthr)))**2

    def __call__(self, pre_current, post_voltage):
        # (N_out, 1) x (1, N_in) -> (N_out, N_in) 
        pre_ = np.expand_dims(pre_current, axis=0)
        post_ = np.expand_dims(
                self.surrogate_derivative_fastsigmoid(post_voltage), 
                axis=1)

        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*(1-self.dt/self.td) + (post_ @ pre_)/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

#################
## モデルの定義  ###
#################

dt = 1e-4; T = 0.5; nt = round(T/dt)

t_weight_update = 0.5 #重みの更新時間
nt_b = round(t_weight_update/dt) #重みの更新ステップ

num_iter = 200 # 学習のイテレーション数

N_in = 50 # 入力ユニット数
N_mid = 4 # 中間ユニット数
N_out = 1 # 出力ユニット数

# 入力(x)と教師信号(y)の定義
fr_in = 10 # 入力のPoisson発火率 (Hz)
x = np.where(np.random.rand(nt, N_in) < fr_in * dt, 1, 0)
y = np.zeros((nt, N_out)) 
y[int(nt/10)::int(nt/5), :] = 1 # T/5に1回発火

# モデルの定義
neurons_1 = CurrentBasedLIF(N_mid, dt=dt)
neurons_2 = CurrentBasedLIF(N_out, dt=dt)
delay_conn1 = DelayConnection(N_in, delay=8e-4)
delay_conn2 = DelayConnection(N_mid, delay=8e-4)
synapses_1 = DoubleExponentialSynapse(N_in, dt=dt, td=1e-2, tr=5e-3)
synapses_2 = DoubleExponentialSynapse(N_mid, dt=dt, td=1e-2, tr=5e-3)
es = ErrorSignal(N_out)
et1 = EligibilityTrace(N_in, N_mid)
et2 = EligibilityTrace(N_mid, N_out)

connect_1 = FullConnection(N_in, N_mid, 
                           initW=0.1*np.random.rand(N_mid, N_in))
connect_2 = FullConnection(N_mid, N_out, 
                           initW=0.1*np.random.rand(N_out, N_mid))
B = np.random.rand(N_mid, N_out)

r0 = 1e-3
gamma = 0.7

# 記録用配列
current_arr = np.zeros((N_mid, nt))
voltage_arr = np.zeros((N_out, nt))
error_arr = np.zeros((N_out, nt))
lambda_arr = np.zeros((N_out, N_mid, nt))
dw_arr = np.zeros((N_out, N_mid, nt))
cost_arr = np.zeros(num_iter)

#################
## シミュレーション ###
#################
for i in tqdm(range(num_iter)):
    if i%15 == 0:
        r0 /= 2 # 重み減衰
    
    # 状態の初期化
    neurons_1.initialize_states()
    neurons_2.initialize_states()
    synapses_1.initialize_states()
    synapses_2.initialize_states()
    delay_conn1.initialize_states()
    delay_conn2.initialize_states()
    es.initialize_states()
    et1.initialize_states()
    et2.initialize_states()
    
    m1 = np.zeros((N_mid, N_in))
    m2 = np.zeros((N_out, N_mid))
    v1 = np.zeros((N_mid, N_in))
    v2 = np.zeros((N_out, N_mid))
    cost = 0
    count = 0
    
    # one iter.
    for t in range(nt):
        # Feed-forward
        c1 = synapses_1(delay_conn1(x[t])) # input current
        h1 = connect_1(c1)
        s1 = neurons_1(h1) # spike of mid neurons
        
        c2 = synapses_2(delay_conn2(s1))
        h2 = connect_2(c2)
        s2 = neurons_2(h2)
        
        # Backward(誤差の伝搬)
        e2 = np.expand_dims(es(s2, y[t]), axis=1) / N_out
        e1 = connect_2.backward(e2) / N_mid
        e1 = B @ e2 / N_mid

        # コストの計算
        cost += np.sum(e2**2)
        
        lambda2 = et2(c2, neurons_2.v_)
        lambda1 = et1(c1, neurons_1.v_)
        
        g2 = e2 * lambda2
        g1 = e1 * lambda1
        
        v1 = np.maximum(gamma*v1, g1**2)
        v2 = np.maximum(gamma*v2, g2**2)
        
        m1 += g1
        m2 += g2
    
        count += 1
        if count == nt_b:
            # 重みの更新
            lr1 = r0/np.sqrt(v1+1e-8)
            lr2 = r0/np.sqrt(v2+1e-8)
            dW1 = np.clip(lr1*m1*dt, -1e-3, 1e-3)
            dW2 = np.clip(lr2*m2*dt, -1e-3, 1e-3)
            connect_1.W = np.clip(connect_1.W+dW1, -0.1, 0.1)
            connect_2.W = np.clip(connect_2.W+dW2, -0.1, 0.1)
            
            # リセット
            m1 = np.zeros((N_mid, N_in))
            m2 = np.zeros((N_out, N_mid))
            v1 = np.zeros((N_mid, N_in))
            v2 = np.zeros((N_out, N_mid))
            count = 0
            
        # 保存
        if i == num_iter-1:
            current_arr[:, t] = c2
            voltage_arr[:, t] = neurons_2.v_
            error_arr[:, t] = e2
            lambda_arr[:, :, t] = lambda2
    
    cost_arr[i] = cost
    print("\n　cost:", cost)

#################
#### 結果表示 ####
#################
def hide_ticks(): #上と右の軸を表示しないための関数
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')

t = np.arange(nt)*dt*1e3
plt.figure(figsize=(8, 10))
plt.subplot(6,1,1)
plt.plot(t, voltage_arr[0], color="k")
plt.ylabel('Membrane\n potential (mV)') 
hide_ticks()
plt.tight_layout()

plt.subplot(6,1,2)
plt.plot(t, et1.surrogate_derivative_fastsigmoid(u=voltage_arr[0]),
         color="k")
plt.ylabel('Surrogate derivative')
hide_ticks()
plt.tight_layout()

plt.subplot(6,1,3)
plt.plot(t, error_arr[0], color="k")
plt.ylabel('Error')
hide_ticks()
plt.tight_layout()

plt.subplot(6,1,4)
plt.plot(t, lambda_arr[0, 0], color="k")
plt.ylabel('$\lambda$')
hide_ticks()
plt.tight_layout()


plt.subplot(6,1,5)
plt.plot(t, current_arr[0], color="k")
plt.ylabel('Input current (pA)')
hide_ticks()
plt.tight_layout()

plt.subplot(6,1,6)
for i in range(N_in):    
    plt.plot(t, x[:, i]*(i+1), 'ko', markersize=2,
             rasterized=True)
hide_ticks()
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index') 
plt.xlim(0, t.max())
plt.ylim(0.5, N_in+0.5)
plt.tight_layout()
plt.savefig("super_spike.svg")
plt.show()

plt.figure(figsize=(4, 3))
plt.plot(cost_arr, color="k")
plt.title('Cost')
plt.xlabel('Iter')
plt.ylabel('Cost') 
hide_ticks()
plt.tight_layout()
plt.savefig("super_spike_cost_fa.pdf")
plt.show()