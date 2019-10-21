# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

import chainer
from LIF_WTA_STDP_MNIST import online_load_and_encoding_dataset, prediction
from LIF_WTA_STDP_MNIST import DiehlAndCook2015Network

#################
####  Main   ####
#################
# 350ms画像入力、150ms入力なしでリセットさせる(膜電位の閾値以外)
dt = 1e-3 # タイムステップ(sec)
t_inj = 0.350 # 刺激入力時間(sec)
t_blank = 0.150 # ブランク時間(sec)
nt_inj = round(t_inj/dt)
nt_blank = round(t_blank/dt)

n_neurons = 100 #興奮性/抑制性ニューロンの数
n_labels = 10 #ラベル数

n_train = 1000 # 訓練データの数
update_nt = nt_inj # STDP則による重みの更新間隔

_, test = chainer.datasets.get_mnist() # ChainerによるMNISTデータの読み込み
labels = np.array([test[i][1] for i in range(n_train)]) # ラベルの配列

# ネットワークの定義
results_save_dir = "./LIF_WTA_STDP_MNIST_results/" # 結果が保存されているディレクトリ

network = DiehlAndCook2015Network(n_in=784, n_neurons=n_neurons,
                                  wexc=2.25, winh=0.85, dt=dt)
network.initialize_states() # ネットワークの初期化
network.input_conn.W = np.load(results_save_dir+"weight.npy")
network.exc_neurons.theta = np.load(results_save_dir+"exc_neurons_theta.npy")
network.exc_neurons.theta_plus = 0

spikes = np.zeros((n_train, n_neurons)).astype(np.uint8) #スパイクを記録する変数
blank_input = np.zeros(784) # ブランク入力
init_max_fr = 32 # 初期のポアソンスパイクの最大発火率

#################
##　Simulation  ##
#################
for i in tqdm(range(n_train)):
    max_fr = init_max_fr
    while(True):
        # 入力スパイクをオンラインで生成
        input_spikes = online_load_and_encoding_dataset(test, i, dt,
                                                        nt_inj, max_fr)
        spike_list = [] # サンプルごとにスパイクを記録するリスト
        # 画像刺激の入力
        for t in range(nt_inj):
            s_exc = network(input_spikes[t], stdp=False)
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


# 入力サンプルのラベルを予測する
assignments = np.load(results_save_dir+"assignments.npy")
predicted_labels = prediction(spikes, assignments, n_labels)

# 訓練精度を計算
accuracy = np.mean(np.where(labels==predicted_labels, 1, 0)).astype(np.float16)
print("Test accuracy :", accuracy)
