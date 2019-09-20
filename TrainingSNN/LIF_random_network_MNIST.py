import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import chainer

from Models.Neurons import CurrentBasedLIF, DiehlAndCook2015LIF
from Models.Synapses import SingleExponentialSynapse

#np.random.seed(seed=0)

dt = 1e-3; T = 0.35; nt = round(T/dt) # シミュレーション時間
num_in = 784; num_out = 10 # 入力 / 出力ニューロンの数

max_fr = 10

train, _ = chainer.datasets.get_mnist()
input_spikes = np.zeros((nt, 784)) # 784=28x28

i = 0

fr = max_fr * np.repeat(np.expand_dims(train[i][0],
                                       axis=0), nt, axis=0)
input_spikes = np.where(np.random.rand(nt, 784) < fr*dt, 1, 0)

input_spikes = input_spikes.astype(np.float32)
plt.imshow(np.reshape(np.sum(input_spikes, axis=0), (28, 28)))

fr_in = 32 # 入力のポアソンスパイクの発火率(Hz)
x = np.where(np.random.rand(nt, num_in) < fr_in * dt, 1, 0)
W = 1*np.random.rand(num_out, num_in) # ランダムな結合重み

# モデル
"""
neurons = CurrentBasedLIF(N=num_out, dt=dt, tref=5e-3,
                          tc_m=1e-2, vrest=-65, vreset=-60,
                          vthr=-40, vpeak=30)
"""
neurons = DiehlAndCook2015LIF(num_out, dt=dt, tref=5e-3, tc_m=1e-1,
                              vrest=-65, vreset=-65, init_vthr=-52,
                              vpeak=20, thr_plus=0.05, tc_thr=1e4,
                              e_exc=0, e_inh=-100)

synapses = SingleExponentialSynapse(N=num_out, dt=dt,td=2e-2)

# 記録用配列
current = np.zeros((num_out, nt))
voltage = np.zeros((num_out, nt))

# シミュレーション
neurons.initialize_states() # 状態の初期化
for t in tqdm(range(nt)):
    # 更新
    #I = synapses(np.dot(W, x[t]))
    I = synapses(np.dot(W, input_spikes[t]))
    s = neurons(I, 17)

    # 記録
    current[:, t] = I
    voltage[:, t] = neurons.v_
    
# 結果表示
t = np.arange(nt)*dt
plt.figure(figsize=(7, 6))
plt.subplot(3,1,1)
plt.plot(t, voltage[0], color="k")
plt.xlim(0, T)
plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()

plt.subplot(3,1,2)
plt.plot(t, current[0], color="k")
plt.xlim(0, T)
plt.ylabel('Synaptic current (pA)') 
plt.tight_layout()
plt.show()
"""
plt.subplot(3,1,3)
for i in range(num_in):    
    plt.plot(t, input_spikes[:, i]*(i+1), 'ko', markersize=2)
plt.xlabel('Time (s)')
plt.ylabel('Neuron index') 
plt.xlim(0, T)
plt.ylim(0.5, num_in+0.5)
plt.tight_layout()
#plt.savefig('LIF_random_network.pdf')
plt.show()
"""