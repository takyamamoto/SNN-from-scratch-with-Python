# 一定入力

import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import cupy as cp
import numpy as np
import chainer.cuda

cp.random.seed(seed=0)

 
class LIF:
    def __init__(self, N, dt=1e-4, tref=5e-3, tc_m=1e-2,
                 vrest=-60, vreset=-60, vthr=-50, vpeak=20):
        """
        Leaky integrate-and-fire model.
        
        Args:
            N (int): Number of neurons.
            dt (float): Simulation time step in seconds.
            tc_m (float):Membrane time constant in seconds. 
            tref (float):Refractory time constant in seconds.
            vreset (float):Reset voltage.
            vthr (float):Threshold voltage.
            vpeak (float):Peak voltage.
        """
        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m 
        self.vrest = vrest
        self.vreset = vreset
        self.vthr = vthr
        self.vpeak = vpeak
        
        self.v = None
        self.v_ = None
        self.tlast = None
    
    def initialize_state(self):
        self.v = self.vreset*cp.ones(self.N) # + np.random.rand(self.N)*(self.vthr-self.vreset) 
        self.tlast = 0
        
    def __call__(self, I, t):
        if self.v is None:
            self.initialize_state()
        
        #膜電位の更新
        dv = ((self.dt*t) > (self.tlast + self.tref))*(self.vrest - self.v + I) / self.tc_m #Voltage equation with refractory period 
        v = self.v + self.dt*dv
        
        #発火の確認
        s = 1*(v>=self.vthr) #発火時は1, その他は0の出力
        self.tlast = self.tlast + (self.dt*t - self.tlast)*s #最後の発火時の更新
        
        v = v + (self.vpeak - v)*s #閾値を超えると膜電位をvpeakにする
        self.v_ = v #発火時の電位も含めて記録するための変数
        
        #発火時に膜電位をリセット
        self.v = v + (self.vreset - v)*s
        
        return s
    
class DoubleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.N = N
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = cp.zeros(N)
        self.hr = cp.zeros(N)
    
    def initialize_state(self):
        self.r = cp.zeros(self.N)
        self.hr = cp.zeros(self.N)
        
    def __call__(self, spike):
        r = self.r*math.exp(-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*math.exp(-self.dt/self.td) + spike/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

class SingleExponentialSynapse:
    def __init__(self, N, dt=1e-4, td=5e-3):
        """
        Args:
            td (float):Synaptic decay time
        """
        self.dt = dt
        self.td = td
        self.r = cp.zeros(N)
    
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.td) + spike/self.td
        self.r = r
        return r

class Linear:
    def __init__(self, N_in, N_out, init_W=None):
        """
        Linear connection.
        """
        if init_W is not None:
            self.W = init_W
        else:
            self.W = 0.1*cp.random.rand(N_out, N_in)
    
    def backward(self, x):
        return self.W.T @ x
    
    def __call__(self, x):
        return self.W @ x
    
class ErrorSignal:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = cp.zeros(N)
        self.hr = cp.zeros(N)
        t1 = td*tr/(td-tr)
        t2 = tr/td
        self.b = t2**(t1/td) - t2**(t1/tr)
        
    def __call__(self, output_spike, target_spike):
        r = self.r*math.exp(-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*math.exp(-self.dt/self.td) + (target_spike - output_spike)/(60*self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

def surrogate_derivative_fastsigmoid(u, beta=1, vthr=-50):
    return 1 / (1 + cp.abs(beta*(u-vthr)))**2

class EligibilityTrace:
    def __init__(self, N_in, N_out, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.dt = dt
        self.td = td
        self.tr = tr
        self.r = cp.zeros((N_out, N_in))
        self.hr = cp.zeros((N_out, N_in))
    
    def __call__(self, pre, post):
        # (N_out, 1) x (1, N_in) -> (N_out, N_in) 
        pre_ = cp.expand_dims(pre, axis=0)
        post_ = cp.expand_dims(post, axis=1)

        r = self.r*math.exp(-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*math.exp(-self.dt/self.td) + (post_ @ pre_)/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

class DelayConnection:
    def __init__(self, N, delay, dt=1e-4):
        """
        Args:
            delay (float): Delay time
        """
        nt_delay = round(delay/dt) # 遅延のステップ数
        self.state = cp.zeros((N, nt_delay))
        
    def __call__(self, x):
        out = self.state[:, -1] # 出力
        
        self.state[:, 1:] = self.state[:, :-1] # 配列をずらす
        self.state[:, 0] = x # 入力
        
        return out

dt = 1e-4
T = 0.5
nt = round(T/dt)

t_weight_update = 0.5
nt_b = round(t_weight_update/dt)

num_iter = 10

N_in = 50
N_mid = 4
N_out = 2

fr_in = 10 # input poission spike firing rate (Hz)

x = cp.where(cp.random.rand(nt, N_in) < fr_in * dt, 1, 0)
y = cp.zeros((nt, N_out))
y[int(nt/10)::int(nt/5), 0] = 1
y[int(nt/10)::int(nt/5), 1] = 1
#y = np.where(np.random.rand(nt, N_out) < fr_in * dt, 1, 0)

# Initialization
spike_arr = cp.zeros((N_out, nt))
c_arr = cp.zeros((N_in, nt))
v_arr = cp.zeros((N_out, nt))
e_arr = cp.zeros((N_out, nt))

delay_conn1 = DelayConnection(N_in, delay=8e-4)
delay_conn2 = DelayConnection(N_mid, delay=8e-4)
neurons_1 = LIF(N_mid, dt=dt)
neurons_2 = LIF(N_out, dt=dt)
synapses_1 = DoubleExponentialSynapse(N_in, dt=dt, td=1e-2, tr=5e-3)
synapses_2 = DoubleExponentialSynapse(N_mid, dt=dt, td=1e-2, tr=5e-3)
es = ErrorSignal(N_out)
et1 = EligibilityTrace(N_in, N_mid)
et2 = EligibilityTrace(N_mid, N_out)

connect_1 = Linear(N_in, N_mid)
connect_2 = Linear(N_mid, N_out)

#trace_1 = SingleExponentialSynapse(N_mid)
#trace_2 = SingleExponentialSynapse(N_out)

cost_arr = cp.zeros(num_iter)
r0 = 1e-4
gamma = math.exp(-1/10)
# Simulation
for i in tqdm(range(num_iter)):
    if i%25 == 0:
        r0 /= 2
    neurons_1.initialize_state()
    neurons_2.initialize_state()
    synapses_1.initialize_state()
    synapses_2.initialize_state()
    
    m1 = cp.zeros((N_mid, N_in))
    m2 = cp.zeros((N_out, N_mid))
    v1 = cp.zeros((N_mid, N_in))
    v2 = cp.zeros((N_out, N_mid))
    cost = 0
    count = 0
    for t in range(nt):
        # forward
        c1 = synapses_1(delay_conn1(x[t])) # input current
        h1 = connect_1(c1)
        s1 = neurons_1(h1, t) # spike of mid neurons
        
        c2 = synapses_2(delay_conn2(s1)) # input current
        h2 = connect_2(c2)
        s2 = neurons_2(h2, t)
        
        #xi1 = np.expand_dims(trace_1(s1**4), 1)
        #xi2 = np.expand_dims(trace_2(s2**4), 1)
        
        # error
        e2 = cp.expand_dims(es(s2, y[t]), axis=1)
        e1 = connect_2.backward(e2)
        
        cost += cp.sum(e2**2)
        
        lam2 = et2(c2, surrogate_derivative_fastsigmoid(neurons_2.v_))
        lam1 = et1(c1, surrogate_derivative_fastsigmoid(neurons_1.v_))
        
        g2 = e2 * lam2 #- xi2*connect_2.W
        g1 = e1 * lam1 #- xi1*connect_1.W
        
        v1 = cp.maximum(gamma*v1, g1**2)
        v2 = cp.maximum(gamma*v2, g2**2)
        lr1 = r0/cp.sqrt(v1+1e-8)
        lr2 = r0/cp.sqrt(v2+1e-8)
        
        m1 += g1
        m2 += g2
    
        count += 1
        if count == nt_b:
            # Weight update
            #dW1 = lr1*m1*dt
            #dW2 = lr2*m2*dt
            dW1 = cp.clip(lr1*m1*dt, -1e-3, 1e-3)
            dW2 = cp.clip(lr2*m2*dt, -1e-3, 1e-3)
            connect_1.W = cp.clip(connect_1.W+dW1, -0.1, 0.1)
            connect_2.W = cp.clip(connect_2.W+dW2, -0.1, 0.1)
            
            m1 = cp.zeros((N_mid, N_in))
            m2 = cp.zeros((N_out, N_mid))
            v1 = cp.zeros((N_mid, N_in))
            v2 = cp.zeros((N_out, N_mid))

            count = 0
            
        # Save
        if i == num_iter-1:
            spike_arr[:,t] = s2
            c_arr[:, t] = h2
            v_arr[:, t] = neurons_2.v_
            e_arr[:, t] = e2
    
    cost_arr[i] = cost
    print("\n")
    print("cost:", cost)

# Plot

spike_arr = chainer.cuda.to_cpu(spike_arr)
c_arr = chainer.cuda.to_cpu(c_arr)
v_arr = chainer.cuda.to_cpu(v_arr)
e_arr = chainer.cuda.to_cpu(e_arr)
cost_arr = chainer.cuda.to_cpu(cost_arr)

t = np.arange(nt)*dt
plt.figure(figsize=(6, 3))
plt.plot(t, v_arr[0])
plt.title('LIF neuron1')
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()
plt.show()

t = np.arange(nt)*dt
plt.figure(figsize=(6, 3))
plt.plot(t, v_arr[1])
plt.title('LIF neuron2')
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(t, c_arr[0])
plt.title('Input current')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(t, e_arr[0])
plt.title('Error')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

t = np.arange(nt)*dt
plt.figure(figsize=(6, 3))
plt.plot(t, surrogate_derivative_fastsigmoid(v_arr[0]))
plt.title('surrogate derivative')
plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 3))
plt.plot(cost_arr)
plt.title('Cost')
plt.xlabel('Iter')
plt.ylabel('Cost') 
plt.tight_layout()
plt.show()

"""
t = np.arange(nt)*dt
plt.figure(figsize=(6, 5))
for i in range(N):    
    plt.plot(t, spike_arr[i]*(i+1), 'ko', markersize=2)
plt.title('LIF neuron')
plt.xlabel('Time (s)')
plt.ylim(0.5, N+0.5)
#plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()

#plt.savefig('LIF_neuron.png')
#plt.show()
"""
