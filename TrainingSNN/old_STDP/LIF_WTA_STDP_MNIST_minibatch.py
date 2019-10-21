# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:59:44 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from numba import jit

import chainer
#import os 

from Models.Neurons import ConductanceBasedLIF, DiehlAndCook2015LIF
from Models.Synapses import SingleExponentialSynapse
from Models.Connections import FullConnection, DelayConnection


np.random.seed(seed=0)
def load_and_encoding_dataset(dataset, n_datas, dt, n_time, max_fr=32):
    input_spikes = np.zeros((n_datas, n_time, 784)) # 784=28x28
    labels = np.zeros(n_datas)
    for i in tqdm(range(n_datas)):
        fr_tmp = max_fr*196/np.sum(np.heaviside(dataset[i][0],0))
        fr = fr_tmp*np.repeat(np.expand_dims(np.heaviside(dataset[i][0],0),
                                               axis=0), n_time, axis=0)
        input_spikes[i] = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
        
    input_spikes = input_spikes.astype(np.int16) #.astype(np.float32)
    #plt.imshow(np.reshape(np.sum(input_spikes[0], axis=0), (28, 28)))
    #plt.show()
    labels = labels.astype(np.int8)
    
    return input_spikes, labels

def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32):
    fr_tmp = max_fr*196/np.sum(np.heaviside(dataset[i][0],0))
    fr = fr_tmp*np.repeat(np.expand_dims(np.heaviside(dataset[i][0],0),
                                         axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
    input_spikes = input_spikes.astype(np.int16) #.astype(np.float32)
    return input_spikes

# ラベルの割り当て
def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):
    """
    Assign labels to the neurons based on highest average spiking activity.

    spikes (n_samples, time, n_neurons) : A single layer's spiking activity.
    labels (n_samples,) : data labels corresponding to spiking activity.
    n_labels (int)      : The number of target labels in the data.
    rates (n_neurons, n_labels) : If passed, these represent spike rates from a previous ``assign_labels()`` call.
    alpha (float): Rate of decay of label assignments.
    return: Tuple of class assignments, per-class spike proportions, and per-class firing rates.
    """
    n_neurons = spikes.shape[2] 
    
    if rates is None:        
        rates = np.zeros((n_neurons, n_labels)).astype(np.float32)
    
    # 時間の軸でスパイク数の和を取る
    n_spikes = np.sum(spikes, axis=1) # (n_samples, n_neurons)
    #print(n_spikes)
    for i in range(n_labels):
        # サンプル内の同じラベルの数を求める
        n_labeled = np.sum(labels == i).astype(np.int16)
    
        if n_labeled > 0:
            # label == iのサンプルのインデックスを取得
            indices = np.where(labels == i)[0]
            
            # label == iに対する各ニューロンごとの平均発火率を計算(前回の発火率との移動平均)
            rates[:, i] = alpha*rates[:, i] + (np.sum(n_spikes[indices], axis=0)/n_labeled)
    
    sum_rate = np.sum(rates, axis=1)
    sum_rate[sum_rate==0] = 1
    # クラスごとの発火頻度の割合を計算する
    proportions = rates / np.expand_dims(sum_rate, 1) # (n_neurons, n_labels)
    proportions[proportions != proportions] = 0  # Set NaNs to 0
    
    # 最も発火率が高いラベルを各ニューロンに割り当てる
    assignments = np.argmax(proportions, axis=1) #.astype(np.int8) # (n_neuroms, )

    return assignments, proportions, rates

# assign_labelsで割り当てたラベルからサンプルのラベルの予測をする
def prediction(spikes, assignments, n_labels):
    """
    Classify data with the label with highest average spiking activity over all neurons.

    spikes  (n_samples, time, n_neurons) : of a layer's spiking activity.
    assignments (n_neurons,) : neuron label assignments.
    n_labels (int): The number of target labels in the data.
    return: Predictions (n_samples,) resulting from the "all activity" classification scheme.
    """
        
    n_samples = spikes.shape[0]
    
    # 時間の軸でスパイク数の和を取る
    n_spikes = np.sum(spikes, axis=1)#.astype(np.int8) # (n_samples, n_neurons)
    
    # 各サンプルについて各ラベルの発火率を見る
    rates = np.zeros((n_samples, n_labels)).astype(np.float32)
    
    for i in range(n_labels):
        # 各ラベルが振り分けられたニューロンの数
        n_assigns = np.sum(assignments == i).astype(np.int16)
    
        if n_assigns > 0:
            # 各ラベルのニューロンのインデックスを取得
            indices = np.where(assignments == i)[0]
    
            # 各ラベルのニューロンのレイヤー全体における平均発火数を求める
            rates[:, i] = np.sum(n_spikes[:, indices], axis=1) / n_assigns
    
    # レイヤーの平均発火率が最も高いラベルを出力
    return np.argmax(rates, axis=1).astype(np.int16) # (n_neuroms, )

class DiehlAndCook2015Network:
    def __init__(self, n_in=784, n_neurons=100, wexc=22.5, winh=17.5,
                 dt=1e-3, wmin=0.0, wmax=5e-2, lr=(1e-2, 1e-4),
                 update_nt=350, n_minibatch=20):
        """
        Network of Diehl and Cooks (2015) 
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full
        
        n_in: Number of input neurons. Matches the 1D size of the input data.
        n_neurons: Number of excitatory, inhibitory neurons.
        wexc: Strength of synapse weights from excitatory to inhibitory layer.
        winh: Strength of synapse weights from inhibitory to excitatory layer.
        dt: Simulation time step.
        lr: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        wmin: Minimum allowed weight on input to excitatory synapses.
        wmax: Maximum allowed weight on input to excitatory synapses.
        """
        
        self.dt = dt
        self.lr_p, self.lr_m = lr
        self.wmax = wmax
        self.wmin = wmin

        # Neurons
        
        self.exc_neurons = DiehlAndCook2015LIF(n_neurons, dt=dt, tref=5e-3,
                                               tc_m=1e-1,
                                               vrest=-65, vreset=-65, 
                                               init_vthr=-52,
                                               vpeak=20, theta_plus=0.05,
                                               tc_theta=1e4,
                                               e_exc=0, e_inh=-100)
        self.inh_neurons = ConductanceBasedLIF(n_neurons, dt=dt, tref=2e-3,
                                               tc_m=1e-2,
                                               vrest=-60, vreset=-45,
                                               vthr=-40, vpeak=20,
                                               e_exc=0, e_inh=-85)
        # Synapses
        self.input_synapse = SingleExponentialSynapse(n_in, dt=dt, td=1e-3)
        self.exc_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=1e-3)
        self.inh_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-3)
        
        self.input_synaptictrace = SingleExponentialSynapse(n_in, dt=dt,
                                                            td=2e-2)
        self.exc_synaptictrace = SingleExponentialSynapse(n_neurons, dt=dt,
                                                          td=2e-2)
        
        # Connections
        self.input_conn = FullConnection(n_in, n_neurons,
                                         initW=1e-3*np.random.rand(n_neurons, n_in))
        self.dW = np.zeros((n_neurons, n_in))
        self.exc2inh_W = wexc*np.eye(n_neurons)
        self.inh2exc_W = (winh/n_neurons)*(np.ones((n_neurons, n_neurons)) - np.eye(n_neurons))
        
        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)
        
        self.g_inh = np.zeros(n_neurons)
        self.tcount = 0
        self.n_minibatch = n_minibatch
        self.update_nt = update_nt
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.s_in_ = np.zeros((self.update_nt, n_in)) 
        self.s_exc_ = np.zeros((n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, n_in)) 
        self.x_exc_ = np.zeros((n_neurons, self.update_nt))
        
    def reset_trace(self):
        self.s_in_ = np.zeros((self.update_nt, self.n_in)) 
        self.s_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, self.n_in)) 
        self.x_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.tcount = 0
    
    def initialize_states(self):
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.input_synapse.initialize_states()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()
        
    def __call__(self, s_in, stdp=True, update=False):
        c_in = self.input_synapse(s_in)
        x_in = self.input_synaptictrace(s_in)
        g_in = self.input_conn(c_in)

        s_exc = self.exc_neurons(self.delay_input(g_in), self.g_inh*0.1)
        c_exc = self.exc_synapse(s_exc)
        g_exc = np.dot(self.exc2inh_W, c_exc)
        x_exc = self.exc_synaptictrace(s_exc)
        
        
        s_inh = self.inh_neurons(self.delay_exc2inh(g_exc), 0)
        c_inh = self.inh_synapse(s_inh)
        self.g_inh = np.dot(self.inh2exc_W, c_inh)

        if stdp:
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] = s_exc
            self.x_in_[self.tcount] = x_in 
            self.x_exc_[:, self.tcount] = x_exc
            self.tcount += 1
            
            if self.tcount >= self.update_nt:
                W = np.copy(self.input_conn.W)
                self.dW += self.lr_p*(self.wmax - W)*np.dot(self.s_exc_, self.x_in_)
                self.dW -= self.lr_m*W*np.dot(self.x_exc_, self.s_in_)
                self.reset_trace()
                
                # Online STDP
                if update:                    
                    clipped_dW = np.clip(self.dW / self.n_minibatch, -1e-1, 1e-1)
                    W = np.clip(W + clipped_dW,
                                self.wmin, self.wmax)
                    # postに投射される重みが均一になるようにする
                    W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                    W_abs_sum[W_abs_sum == 0] = 1.0
                    W *= self.wmax / W_abs_sum
                    self.input_conn.W = W
                    self.dW = np.zeros((self.n_neurons, self.n_in))

        return s_exc
        

# 350ms画像入力、150ms入力なしでリセットさせる(膜電位の閾値以外)。
dt = 1e-3 # sec
t_inj = 0.35 # 0.350 # sec
t_blank = 0.15 # 0.150 # sec
nt_inj = round(t_inj/dt)
nt_blank = round(t_blank/dt)

n_neurons = 100
n_labels = 10
n_epoch = 20

n_train = 1000
update_nt = nt_inj # traceの更新時間
n_minibatch = 20 # minibatch数
train, _ = chainer.datasets.get_mnist()
labels = np.array([train[i][1] for i in range(n_train)])

network = DiehlAndCook2015Network(n_in=784, n_neurons=n_neurons,
                                  wexc=22.5, winh=17.5,
                                  dt=dt, wmin=0.0, wmax=1,
                                  lr=(1e-2, 1e-4), update_nt=update_nt,
                                  n_minibatch=n_minibatch)

network.initialize_states()
spikes = np.zeros((n_train, nt_inj, n_neurons)).astype(np.int8)
#voltage = np.zeros((n_train, nt_inj, n_neurons)) #.astype(np.int8)
accuracy_all = np.zeros(n_epoch)
blank_input = np.zeros(784)
init_max_fr = 32

for epoch in range(n_epoch):
    for i in tqdm(range(n_train)):
        max_fr = init_max_fr
        b_update = True if i % n_minibatch==0 else False # 更新するかのbool
        while(True):
            input_spikes = online_load_and_encoding_dataset(train, i, dt,
                                                            nt_inj, max_fr)
            for t in range(nt_inj):
                s_exc = network(input_spikes[t], stdp=True, update=b_update)
                spikes[i, t] = s_exc
                
            #network.initialize_states()
            for _ in range(nt_blank):
                _ = network(blank_input, stdp=False)

            num_spikes_exc = np.sum(spikes[i])
            if num_spikes_exc >= 5:
                break
            else:
                #print("max_fr", max_fr)
                max_fr += 16
        
        """
        t = np.arange(nt_inj)*dt
        plt.figure(figsize=(7, 6))
        plt.plot(t, voltage[0, :, 0], color="k")
        plt.xlim(0, t_inj)
        plt.ylabel('Membrane potential (mV)') 
        plt.tight_layout()
        plt.plot()
        """
        
    if epoch == 0:
        assignments, proportions, rates = assign_labels(spikes, labels, n_labels)
    else:
        #print(rates)
        assignments, proportions, rates = assign_labels(spikes, labels,
                                                        n_labels, rates)
    #print(assignments)
    sum_nspikes = np.sum(np.sum(spikes, axis=1), axis=1)
    mean_nspikes = np.mean(sum_nspikes).astype(np.float16)
    print("Ave. spikes:", mean_nspikes)
    print("Min. spikes:", sum_nspikes.min())
    print("Max. spikes:", sum_nspikes.max())
    #if mean_nspikes < 10:
    #    init_max_fr += 5
    predicted_labels = prediction(spikes, assignments, n_labels)
    
    accuracy = np.mean(np.where(labels==predicted_labels, 1, 0)).astype(np.float16)
    print("epoch :", epoch, " accuracy :", accuracy)
    accuracy_all[epoch] = accuracy

plt.plot(accuracy_all)
plt.xlabel("Epoch")
plt.ylabel("Train accuracy")
plt.savefig("accuracy.svg")
plt.show()
np.save("weight.npy", network.input_conn.W)
#W = np.load("weight.npy")
#print(W.shape)

