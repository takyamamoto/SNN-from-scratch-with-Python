# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
import os 

from Models.Neurons import ConductanceBasedLIF, DiehlAndCook2015LIF
from Models.Synapses import SingleExponentialSynapse
from Models.Connections import FullConnection, DelayConnection

np.random.seed(seed=0)

#################
####  Utils  ####
#################
# 画像をポアソンスパイク列に変換
def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32,
                                     norm=180):
    fr_tmp = max_fr*norm/np.sum(dataset[i][0])
    fr = fr_tmp*np.repeat(np.expand_dims(dataset[i][0],
                                         axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)

    return input_spikes

# ラベルの割り当て
def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):
    """
    Assign labels to the neurons based on highest average spiking activity.
    
    Args:
        spikes (n_samples, n_neurons) : A single layer's spiking activity.
        labels (n_samples,) : Data labels corresponding to input samples.
        n_labels (int)      : The number of target labels in the data.
        rates (n_neurons, n_labels) : If passed, these represent spike rates
                                      from a previous ``assign_labels()`` call.
        alpha (float): Rate of decay of label assignments.
    return: Class assignments, per-class spike proportions, and per-class firing rates.
    """
    n_neurons = spikes.shape[1] 
    
    if rates is None:        
        rates = np.zeros((n_neurons, n_labels)).astype(np.float32)
    
    # 時間の軸でスパイク数の和を取る
    for i in range(n_labels):
        # サンプル内の同じラベルの数を求める
        n_labeled = np.sum(labels == i).astype(np.int16)
    
        if n_labeled > 0:
            # label == iのサンプルのインデックスを取得
            indices = np.where(labels == i)[0]
            
            # label == iに対する各ニューロンごとの平均発火率を計算(前回の発火率との移動平均)
            rates[:, i] = alpha*rates[:, i] + (np.sum(spikes[indices], axis=0)/n_labeled)
    
    sum_rate = np.sum(rates, axis=1)
    sum_rate[sum_rate==0] = 1
    # クラスごとの発火頻度の割合を計算する
    proportions = rates / np.expand_dims(sum_rate, 1) # (n_neurons, n_labels)
    proportions[proportions != proportions] = 0  # Set NaNs to 0
    
    # 最も発火率が高いラベルを各ニューロンに割り当てる
    assignments = np.argmax(proportions, axis=1).astype(np.uint8) # (n_neurons,)

    return assignments, proportions, rates

# assign_labelsで割り当てたラベルからサンプルのラベルの予測をする
def prediction(spikes, assignments, n_labels):
    """
    Classify data with the label with highest average spiking activity over all neurons.

    Args:
        spikes  (n_samples, n_neurons) : A layer's spiking activity.
        assignments (n_neurons,) : Neuron label assignments.
        n_labels (int): The number of target labels in the data.
    return: Predictions (n_samples,)
    """
        
    n_samples = spikes.shape[0]
    
    # 各サンプルについて各ラベルの発火率を見る
    rates = np.zeros((n_samples, n_labels)).astype(np.float32)
    
    for i in range(n_labels):
        # 各ラベルが振り分けられたニューロンの数
        n_assigns = np.sum(assignments == i).astype(np.uint8)
    
        if n_assigns > 0:
            # 各ラベルのニューロンのインデックスを取得
            indices = np.where(assignments == i)[0]
    
            # 各ラベルのニューロンのレイヤー全体における平均発火数を求める
            rates[:, i] = np.sum(spikes[:, indices], axis=1) / n_assigns
    
    # レイヤーの平均発火率が最も高いラベルを出力
    return np.argmax(rates, axis=1).astype(np.uint8) # (n_samples, )


#################
####  Model  ####
#################
class DiehlAndCook2015Network:
    def __init__(self, n_in=784, n_neurons=100, wexc=2.25, winh=0.875,
                 dt=1e-3, wmin=0.0, wmax=5e-2, lr=(1e-2, 1e-4),
                 update_nt=100):
        """
        Network of Diehl and Cooks (2015) 
        https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full
        
        Args:
            n_in: Number of input neurons. Matches the 1D size of the input data.
            n_neurons: Number of excitatory, inhibitory neurons.
            wexc: Strength of synapse weights from excitatory to inhibitory layer.
            winh: Strength of synapse weights from inhibitory to excitatory layer.
            dt: Simulation time step.
            lr: Single or pair of learning rates for pre- and post-synaptic events, respectively.
            wmin: Minimum allowed weight on input to excitatory synapses.
            wmax: Maximum allowed weight on input to excitatory synapses.
            update_nt: Number of time steps of weight updates.
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
                                               theta_max=35,
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
        initW = 1e-3*np.random.rand(n_neurons, n_in)
        self.input_conn = FullConnection(n_in, n_neurons,
                                         initW=initW)
        self.exc2inh_W = wexc*np.eye(n_neurons)
        self.inh2exc_W = (winh/n_neurons)*(np.ones((n_neurons, n_neurons)) - np.eye(n_neurons))
        
        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)
        
        self.norm = 0.1
        self.g_inh = np.zeros(n_neurons)
        self.tcount = 0
        self.update_nt = update_nt
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.s_in_ = np.zeros((self.update_nt, n_in)) 
        self.s_exc_ = np.zeros((n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, n_in)) 
        self.x_exc_ = np.zeros((n_neurons, self.update_nt))
        
    # スパイクトレースのリセット
    def reset_trace(self):
        self.s_in_ = np.zeros((self.update_nt, self.n_in)) 
        self.s_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, self.n_in)) 
        self.x_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.tcount = 0
    
    # 状態の初期化
    def initialize_states(self):
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.input_synapse.initialize_states()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()
        
    def __call__(self, s_in, stdp=True):
        # 入力層
        c_in = self.input_synapse(s_in)
        x_in = self.input_synaptictrace(s_in)
        g_in = self.input_conn(c_in)

        # 興奮性ニューロン層
        s_exc = self.exc_neurons(self.delay_input(g_in), self.g_inh)
        c_exc = self.exc_synapse(s_exc)
        g_exc = np.dot(self.exc2inh_W, c_exc)
        x_exc = self.exc_synaptictrace(s_exc)

        # 抑制性ニューロン層        
        s_inh = self.inh_neurons(self.delay_exc2inh(g_exc), 0)
        c_inh = self.inh_synapse(s_inh)
        self.g_inh = np.dot(self.inh2exc_W, c_inh)

        if stdp:
            # スパイク列とスパイクトレースを記録
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] = s_exc
            self.x_in_[self.tcount] = x_in 
            self.x_exc_[:, self.tcount] = x_exc
            self.tcount += 1

            # Online STDP
            if self.tcount == self.update_nt:
                W = np.copy(self.input_conn.W)
                
                # postに投射される重みが均一になるようにする
                W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                W_abs_sum[W_abs_sum == 0] = 1.0
                W *= self.norm / W_abs_sum
                
                # STDP則
                dW = self.lr_p*(self.wmax - W)*np.dot(self.s_exc_, self.x_in_)
                dW -= self.lr_m*W*np.dot(self.x_exc_, self.s_in_)
                clipped_dW = np.clip(dW / self.update_nt, -5e-2, 5e-2)
                self.input_conn.W = np.clip(W + clipped_dW,
                                            self.wmin, self.wmax)
                self.reset_trace() # スパイク列とスパイクトレースをリセット
        
        return s_exc
        
if __name__ == '__main__':
    # 350ms画像入力、150ms入力なしでリセットさせる(膜電位の閾値以外)
    dt = 1e-3 # タイムステップ(sec)
    t_inj = 0.350 # 刺激入力時間(sec)
    t_blank = 0.150 # ブランク時間(sec)
    nt_inj = round(t_inj/dt)
    nt_blank = round(t_blank/dt)
    
    n_neurons = 100 #興奮性/抑制性ニューロンの数
    n_labels = 10 #ラベル数
    n_epoch = 30 #エポック数
    
    n_train = 10000 # 訓練データの数
    update_nt = nt_inj # STDP則による重みの更新間隔
    
    train, _ = chainer.datasets.get_mnist() # ChainerによるMNISTデータの読み込み
    labels = np.array([train[i][1] for i in range(n_train)]) # ラベルの配列
    
    # ネットワークの定義
    network = DiehlAndCook2015Network(n_in=784, n_neurons=n_neurons,
                                      wexc=2.25, winh=0.875,
                                      dt=dt, wmin=0.0, wmax=0.1,
                                      lr=(1e-2, 1e-4),
                                      update_nt=update_nt)
    
    network.initialize_states() # ネットワークの初期化
    spikes = np.zeros((n_train, n_neurons)).astype(np.uint8) #スパイクを記録する変数
    accuracy_all = np.zeros(n_epoch) # 訓練精度を記録する変数
    blank_input = np.zeros(784) # ブランク入力
    init_max_fr = 32 # 初期のポアソンスパイクの最大発火率
    
    results_save_dir = "./LIF_WTA_STDP_MNIST_results2/" # 結果を保存するディレクトリ
    os.makedirs(results_save_dir, exist_ok=True) # ディレクトリが無ければ作成
    
    #################
    ##　Simulation  ##
    #################
    for epoch in range(n_epoch):
        for i in tqdm(range(n_train)):
            max_fr = init_max_fr
            while(True):
                # 入力スパイクをオンラインで生成
                input_spikes = online_load_and_encoding_dataset(train, i, dt,
                                                                nt_inj, max_fr)
                spike_list = [] # サンプルごとにスパイクを記録するリスト
                # 画像刺激の入力
                for t in range(nt_inj):
                    s_exc = network(input_spikes[t], stdp=True)
                    spike_list.append(s_exc)
                
                spikes[i] = np.sum(np.array(spike_list), axis=0) # スパイク数を記録
                
                # ブランク刺激の入力
                for _ in range(nt_blank):
                    _ = network(blank_input, stdp=False)
    
                num_spikes_exc = np.sum(np.array(spike_list)) # スパイク数を計算
                if num_spikes_exc >= 5: # スパイク数が5より大きければ次のサンプルへ
                    break
                else: # スパイク数が5より小さければ入力発火率を上げて再度刺激
                    max_fr += 16
        
        # ニューロンを各ラベルに割り当てる
        if epoch == 0:
            assignments, proportions, rates = assign_labels(spikes, labels,
                                                            n_labels)
        else:
            assignments, proportions, rates = assign_labels(spikes, labels,
                                                            n_labels, rates)
        print("Assignments:\n", assignments)
        
        # スパイク数の確認(正常に発火しているか確認)
        sum_nspikes = np.sum(spikes, axis=1)
        mean_nspikes = np.mean(sum_nspikes).astype(np.float16)
        print("Ave. spikes:", mean_nspikes)
        print("Min. spikes:", sum_nspikes.min())
        print("Max. spikes:", sum_nspikes.max())
    
        # 入力サンプルのラベルを予測する
        predicted_labels = prediction(spikes, assignments, n_labels)
        
        # 訓練精度を計算
        accuracy = np.mean(np.where(labels==predicted_labels, 1, 0)).astype(np.float16)
        print("epoch :", epoch, " accuracy :", accuracy)
        accuracy_all[epoch] = accuracy
        
        # 学習率の減衰
        network.lr_p *= 0.85
        network.lr_m *= 0.85
        
        # 重みの保存(エポック毎)
        np.save(results_save_dir+"weight_epoch"+str(epoch)+".npy",
                network.input_conn.W)
        
    #################
    ###　 Results  ###
    #################
    plt.figure(figsize=(5,4))
    plt.plot(np.arange(1, n_epoch+1), accuracy_all*100,
             color="k")
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy (%)")
    plt.savefig(results_save_dir+"accuracy.png")
    #plt.show()
    
    np.save(results_save_dir+"assignments.npy", assignments)
    np.save(results_save_dir+"weight.npy", network.input_conn.W)
    np.save(results_save_dir+"exc_neurons_theta.npy",
            network.exc_neurons.theta)
