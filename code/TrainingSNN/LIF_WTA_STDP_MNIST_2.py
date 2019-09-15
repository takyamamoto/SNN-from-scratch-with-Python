# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:59:44 2019

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
import os 

from Models.Neurons import ConductanceBasedLIF, DiehlAndCook2015LIF
from Models.Synapses import SingleExponentialSynapse
from Models.Connections import FullConnection, DelayConnection


np.random.seed(seed=0)

def load_and_encoding_dataset(n_datas, dt, n_time, max_fr=60):
    if os.path.exists("spiking_mnist2.npy"):
        input_spikes = np.load("spiking_mnist.npy")
        labels = np.load("spiking_mnist_labels.npy")
    else:
        train, _ = chainer.datasets.get_mnist()
        input_spikes = np.zeros((n_datas, n_time, 784)) # 784=28x28

        labels = np.zeros(n_datas)
        for i in tqdm(range(n_datas)):
            fr = max_fr * np.repeat(np.expand_dims(train[i][0], axis=0), n_time, axis=0)
            input_spikes[i] = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
            labels[i] = train[i][1]
            
        input_spikes = input_spikes.astype(np.float32)
        # plt.imshow(np.reshape(np.sum(input_spikes[0], axis=0), (28, 28)))
        labels = labels.astype(np.int8)

        np.save("spiking_mnist.npy", input_spikes)
        np.save("spiking_mnist_labels.npy", labels)
    
    return input_spikes, labels

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
    spikes = np.sum(spikes, axis=1) # (n_samples, n_neurons)
    
    for i in range(n_labels):
        # サンプル内の同じラベルの数を求める
        n_labeled = np.sum(labels == i).astype(np.int8)
    
        if n_labeled > 0:
            # label == iのサンプルのインデックスを取得
            indices = np.nonzero(labels == i)[0]
            
            # label == iに対する各ニューロンごとの平均発火率を計算(前回の発火率との移動平均)
            rates[:, i] = alpha*rates[:, i] + (np.sum(spikes[indices], axis=0)/n_labeled)
            
    # クラスごとの発火頻度の割合を計算する
    proportions = rates / np.expand_dims(np.sum(rates, axis=1), 1) # (n_neurons, n_labels)
    proportions[proportions != proportions] = 0  # Set NaNs to 0
    
    # 最も発火率が高いラベルを各ニューロンに割り当てる
    assignments = np.argmax(proportions, axis=1).astype(np.int8) # (n_neuroms, )

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
    spikes = np.sum(spikes, axis=1)#.astype(np.int8) # (n_samples, n_neurons)
    
    # 各サンプルについて各ラベルの発火率を見る
    rates = np.zeros((n_samples, n_labels)).astype(np.float32)
    
    for i in range(n_labels):
        # 各ラベルが振り分けられたニューロンの数
        n_assigns = np.sum(assignments == i).astype(np.int8)
    
        if n_assigns > 0:
            # 各ラベルのニューロンのインデックスを取得
            indices = np.nonzero(assignments == i)[0]
    
            # 各ラベルのニューロンのレイヤー全体における平均発火数を求める
            rates[:, i] = np.sum(spikes[:, indices], axis=1) / n_assigns
    
    # レイヤーの平均発火率が最も高いラベルを出力
    return np.argmax(rates, axis=1).astype(np.int8) # (n_neuroms, )

class DiehlAndCook2015Network:
    def __init__(self, N_in=784, N_neurons=100, wexc=0.0225, winh=0.0175,
                 dt=1e-3, wmin=0.0, wmax=0.001, lr=(1e-4, 1e-2)):
        """
        Network of Diehl and Cooks (2015) 
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full
        
        N_in: Number of input neurons. Matches the 1D size of the input data.
        N_neurons: Number of excitatory, inhibitory neurons.
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
        self.exc_neurons = DiehlAndCook2015LIF(N_neurons, dt=dt, tref=5e-3, tc_m=1e-1,
                                               vrest=-65, vreset=-65, init_vthr=-52,
                                               vpeak=20, thr_plus=0.05, tc_thr=1e4,
                                               e_exc=0, e_inh=-100)
        self.inh_neurons = ConductanceBasedLIF(N_neurons, dt=dt, tref=2e-3, tc_m=1e-2,
                                               vrest=-60, vreset=-45, vthr=-40, vpeak=20,
                                               e_exc=0, e_inh=-85)
        # Synapses
        self.input_synapse = SingleExponentialSynapse(N_in, dt=dt, td=2e-2)
        self.exc_synapse = SingleExponentialSynapse(N_neurons, dt=dt, td=1e-2)
        self.inh_synapse = SingleExponentialSynapse(N_neurons, dt=dt, td=2e-2)
        
        # Connections
        self.input_conn = FullConnection(N_in, N_neurons,
                                         initW=0.0003*np.random.rand(N_neurons, N_in))
        self.exc2inh_conn = FullConnection(N_in, N_neurons,
                                           initW=wexc*np.eye(N_neurons))
        self.inh2exc_conn = FullConnection(N_in, N_neurons,
                                           initW=-winh*(np.ones((N_neurons, N_neurons))-np.eye(N_neurons)))
        
        self.g_inh = 0
        self.tcount = 0
        self.update_nt = 100
        self.N_neurons = N_neurons
        self.s_in_ = np.zeros((self.update_nt, N_neurons)) 
        self.s_exc_ = np.zeros((N_neurons, self.update_nt))
        self.c_in_ = np.zeros((self.update_nt, N_neurons)) 
        self.c_exc_ = np.zeros((N_neurons, self.update_nt))
        
    def reset_trace(self):
        self.s_in_ = np.zeros((self.update_nt, self.N_neurons)) 
        self.s_exc_ = np.zeros((self.N_neurons, self.update_nt))
        self.c_in_ = np.zeros((self.update_nt, self.N_neurons)) 
        self.c_exc_ = np.zeros((self.N_neurons, self.update_nt))
        self.tcount = 0
        
    def __call__(self, s_in, stdp=True):
        c_in = self.input_synapse(s_in)
        g_in = self.input_conn(c_in)
        
        s_exc = self.exc_neurons(g_in, self.g_inh)
        c_exc = self.exc_synapse(s_exc)
        g_exc = self.exc2inh_conn(c_exc)
        
        s_inh = self.inh_neurons(g_exc, 0)
        c_inh = self.inh_synapse(s_inh)
        self.g_inh = self.inh2exc_conn(c_inh)
        
        if stdp:
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] = s_exc
            self.c_in_[self.tcount] = c_in 
            self.c_exc_[:, self.tcount] = c_exc
            self.tcount += 1

            # Online STDP
            if self.tcount > self.update_nt:
                W = self.input_conn.W
                dW = self.lr_p*(self.wmax - W)*np.matmul(self.s_exc_, self.c_in_)
                dW -= self.lr_m*W*np.matmul(self.c_exc_, self.s_in_)
                self.input_conn.W = np.clip(W+dW*self.dt, self.wmin, self.wmax)
                self.reset_trace()
        else:
            if self.tcount > 0:
                self.reset_trace()
    
        return s_exc
        
# dtがsecで, Brian2による実装はmsなので、重みを10^-3倍する
# 350ms画像入力、150ms入力なしでリセットさせる(膜電位の閾値以外)。
dt = 1e-3 # sec
t_inj = 0.10 # 0.350 # sec
t_blank = 0.01 # 0.150 # sec
nt_inj = round(t_inj/dt)
nt_blank = round(t_blank/dt)
n_neurons = 400
n_labels = 10
n_iteration = 30

N_train = 6000
input_spikes, labels = load_and_encoding_dataset(N_train, dt, nt_inj, max_fr=60)
network = DiehlAndCook2015Network(dt=dt, N_in=784, N_neurons=n_neurons)
spikes = np.zeros((N_train, nt_inj, n_neurons)).astype(np.int8)
blank_input = np.zeros(784)

for iter in range(n_iteration):
    for i in tqdm(range(N_train)):
        for t in range(nt_inj):
            s_exc = network(input_spikes[i, t], stdp=True)
            spikes[i, t] = s_exc

        for t in range(nt_blank):
            _ = network(blank_input, stdp=False)

    assignments, proportions, rates = assign_labels(spikes, labels, n_labels)
    predicted_labels = prediction(spikes, assignments, n_labels)
    accuracy = np.mean(np.where(labels==predicted_labels, 1, 0)).astype(np.float32)
    print("iter :", iter, " accuracy :", accuracy)
