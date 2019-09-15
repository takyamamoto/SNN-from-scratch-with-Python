# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

tc_m = 1e-2 # 膜時定数 (s)
R = 1 #膜抵抗 
vthr = 1 # 閾値電位 (mV)
tref = 5e-3 # 不応期 (s)
I_max = 3
I = np.arange(0, I_max, 0.01) #入力電流

rate = 1 / (tref + tc_m*np.log(R*I / (R*I - vthr)))
rate[np.isnan(rate)] = 0 # nan to 0

# 描画
plt.figure(figsize=(4, 3))
plt.plot(I, rate, color="k")
plt.xlabel('Input current (nA)')
plt.ylabel('Firing rate (Hz)') 
plt.xlim(0, I_max)
plt.tight_layout()
plt.savefig('LIF_FI_analytical.pdf')
#plt.show()
