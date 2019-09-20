# -*- coding: utf-8 -*-

import numpy as np

class FullConnection:
    def __init__(self, N_in, N_out, initW=None):
        """
        FullConnection
        """
        if initW is not None:
            self.W = initW
        else:
            self.W = 0.1*np.random.rand(N_out, N_in)
    
    def backward(self, x):
        return np.dot(self.W.T, x) #self.W.T @ x
    
    def __call__(self, x):
        return np.dot(self.W, x) #self.W @ x


class DelayConnection:
    def __init__(self, N, delay, dt=1e-4):
        """
        Args:
            delay (float): Delay time
        """
        self.N = N
        self.nt_delay = round(delay/dt) # 遅延のステップ数
        self.state = np.zeros((N, self.nt_delay))
    
    def initialize_states(self):
        self.state = np.zeros((self.N, self.nt_delay))
    
    def __call__(self, x):
        out = self.state[:, -1] # 出力
        
        self.state[:, 1:] = self.state[:, :-1] # 配列をずらす
        self.state[:, 0] = x # 入力
        
        return out