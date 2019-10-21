# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(seed=0)

# 画像をポアソンスパイク列に変換
def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32,
                                     norm=196):
    fr_tmp = max_fr*norm/np.sum(dataset[i][0])
    fr = fr_tmp*np.repeat(np.expand_dims(dataset[i][0],
                                         axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)

    return input_spikes


import chainer
        
dt = 1e-3; t_inj = 0.350; nt_inj = round(t_inj/dt)
train, _ = chainer.datasets.get_mnist() # ChainerによるMNISTデータの読み込み
input_spikes = online_load_and_encoding_dataset(dataset=train, i=0,
                                                dt=dt, n_time=nt_inj)
# 描画
plt.imshow(np.reshape(np.sum(input_spikes, axis=0), (28, 28)),
           cmap="gray")
plt.savefig("five.pdf")
    