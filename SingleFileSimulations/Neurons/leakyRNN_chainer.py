# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
from chainer import Variable
xp = cuda.cupy
# import numpy as xp

class leakyRNN(chainer.Chain):
    def __init__(self, inp=32, mid=128, alpha=0.2, sigma_rec=0.1):
        super(leakyRNN, self).__init__()
        """ 
        Leaky RNN unit.
        
        Usage:
            >>> network = leakyRNN()
            >>> network.reset_state()
            >>> x = xp.ones((1, 32)).astype(xp.float32)
            >>> y = network(x)
            >>> y.shape
                (1, 128)
        """

        with self.init_scope():
            self.Wx = L.Linear(inp, mid) # feed-forward 
            self.Wr = L.Linear(mid, mid, nobias=True) # recurrent

            self.inp = inp # 入力ユニット数
            self.mid = mid # 出力ユニット数    
            self.alpha = alpha # tau / dt
            self.sigma_rec = sigma_rec # standard deviation of input noise 

    def reset_state(self, r=None):
        self.r = r

    def initialize_state(self, shape):
        self.r = Variable(xp.zeros((shape[0], self.mid),
                                   dtype=self.xp.float32))
        
    def forward(self, x):
        if self.r is None:
            self.initialize_state(x.shape)

        z = self.Wr(self.r) + self.Wx(x)
        if self.sigma_rec is not None:
            z += xp.random.normal(0, self.sigma_rec,
                                  (x.shape[0], self.mid)) # Add noise
        r = (1 - self.alpha)*self.r + self.alpha*F.relu(z)
        
        self.r = r
        return r