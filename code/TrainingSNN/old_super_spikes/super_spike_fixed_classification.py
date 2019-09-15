# 一定入力

import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
 
np.random.seed(seed=0)

 
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
        
        self.v = self.vreset*np.ones(N)
        self.v_ = None
        self.tlast = None
    
    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N)*(self.vthr-self.vreset) 
        else:
            self.v = self.vreset*np.ones(self.N)
        self.tlast = 0
        
    def __call__(self, I, t):
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
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
        self.b = (td/tr)**(td/(tr-td))
    
    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)
        
    def __call__(self, spike):
        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*(1-self.dt/self.td) + spike/(self.tr*self.td)
        
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
        self.r = np.zeros(N)

    def initialize_states(self):
        self.r = np.zeros(self.N)

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
            self.W = 0.1*np.random.rand(N_out, N_in)
    
    def backward(self, x):
        return self.W.T @ x
    
    def __call__(self, x):
        return self.W @ x
    
class ErrorSignal_classification:
    def __init__(self, N, dt=1e-4, td=1e-2, tr=5e-3):
        """
        Args:
            td (float):Synaptic decay time
            tr (float):Synaptic rise time
        """
        self.dt = dt
        self.td = td
        self.tr = tr
        self.N = N
        self.r = np.zeros(N)
        self.hr = np.zeros(N)
        self.b = (td/tr)**(td/(tr-td))
    
    def initialize_states(self):
        self.r = np.zeros(self.N)
        self.hr = np.zeros(self.N)    

    def __call__(self, output_spike, target):
        r = self.r*math.exp(-self.dt/self.tr) + self.hr/self.td*self.dt 
        hr = self.hr*math.exp(-self.dt/self.td) + (target*output_spike)/self.b
        
        self.r = r
        self.hr = hr
        
        return r

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
        self.N_in = N_in
        self.N_out = N_out
        self.r = np.zeros((N_out, N_in))
        self.hr = np.zeros((N_out, N_in))
    
    def initialize_states(self):
        self.r = np.zeros((self.N_out, self.N_in))
        self.hr = np.zeros((self.N_out, self.N_in))
    
    def surrogate_derivative_fastsigmoid(self, u, beta=1, vthr=-50):
        return 1 / (1 + np.abs(beta*(u-vthr)))**2

    def __call__(self, pre_current, post_voltage):
        # (N_out, 1) x (1, N_in) -> (N_out, N_in) 
        pre_ = np.expand_dims(pre_current, axis=0)
        post_ = np.expand_dims(
                self.surrogate_derivative_fastsigmoid(post_voltage), 
                axis=1)

        r = self.r*(1-self.dt/self.tr) + self.hr*self.dt 
        hr = self.hr*(1-self.dt/self.td) + (post_ @ pre_)/(self.tr*self.td)
        
        self.r = r
        self.hr = hr
        
        return r

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

dt = 1e-4
T = 0.2
nt = round(T/dt)

t_weight_update = 0.1
nt_b = round(t_weight_update/dt)

num_iter = 100

# Solving XOR problem
# 2 input, 20 hidden, and 2 output
N_in = 2
N_mid = 20
N_out = 2

"""
sample増やす
"""
# (0.2, 0.2), (1, 0.2), (0.2, 1), (1, 1)
# class: 0, 1, 1, 0
# input poission spike firing rate (Hz)
num_samples = 10
max_fr = 200

inj_time = round(nt/num_samples)

target_class = np.random.randint(0, 2, size=(num_samples))
y = np.zeros((nt, N_out))
for i in range(num_samples):
    y[i*inj_time:(i+1)*inj_time, target_class[i]] = 1

x = np.zeros((nt, N_in))
fr_in = np.zeros((nt, N_in))

for i in range(num_samples):
    if target_class[i] == 0:
        fr_in[i, 0]
        
fr_in = max_fr*np.array([[0.2, 0.2], [1, 0.2], [0.2, 1], [1, 1]])
for i in range(num_samples):
    x[i*inj_time:(i+1)*inj_time] = np.where(
            np.random.rand(inj_time, N_in) < fr_in[i] * dt, 1, 0)


# Initialization
spike_arr = np.zeros((N_out, nt))
c_arr = np.zeros((N_in, nt))
v_arr = np.zeros((N_out, nt))
e_arr = np.zeros((N_out, nt))

neurons_1 = LIF(N_mid, dt=dt)
neurons_2 = LIF(N_out, dt=dt)
delay_conn1 = DelayConnection(N_in, delay=8e-4)
delay_conn2 = DelayConnection(N_mid, delay=8e-4)
synapses_1 = DoubleExponentialSynapse(N_in, dt=dt, td=1e-2, tr=5e-3)
synapses_2 = DoubleExponentialSynapse(N_mid, dt=dt, td=1e-2, tr=5e-3)
es = ErrorSignal_classification(N_out)
et1 = EligibilityTrace(N_in, N_mid)
et2 = EligibilityTrace(N_mid, N_out)

connect_1 = Linear(N_in, N_mid)
connect_2 = Linear(N_mid, N_out)

cost_arr = np.zeros(num_iter)
r0 = 1e-3
gamma = 0.8
#B = np.random.rand(N_mid, N_out)
# Simulation
for i in tqdm(range(num_iter)):
    if i%25 == 0:
        r0 /= 2
    neurons_1.initialize_states()
    neurons_2.initialize_states()
    synapses_1.initialize_states()
    synapses_2.initialize_states()
    delay_conn1.initialize_states()
    delay_conn2.initialize_states()
    es.initialize_states()
    et1.initialize_states()
    et2.initialize_states()
    
    m1 = np.zeros((N_mid, N_in))
    m2 = np.zeros((N_out, N_mid))
    v1 = np.zeros((N_mid, N_in))
    v2 = np.zeros((N_out, N_mid))
    spike_count = np.zeros(N_out)

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
        
        # error
        e2 = np.expand_dims(es(s2, (y[t]-1)), axis=1) / N_out
        
        spike_count += s2
        if t > 0 and t % (inj_time-1) == 0:
            if spike_count[np.where(y[t]==1)] < 1.0:
                es(np.ones(N_out), y[t])
            spike_count = np.zeros(N_out)
            #e2 = np.expand_dims(es(s2, (y[t]-1)), axis=1) / N_out

        
        #e1 = B @ e2
        e1 = connect_2.backward(e2) / N_mid
        
        cost += np.sum(e2**2)
        
        lam2 = et2(c2, neurons_2.v_)
        lam1 = et1(c1, neurons_1.v_)
        
        g2 = e2 * lam2 #- xi2*connect_2.W
        g1 = e1 * lam1 #- xi1*connect_1.W
        
        v1 = np.maximum(gamma*v1, g1**2)
        v2 = np.maximum(gamma*v2, g2**2)
        lr1 = r0/np.sqrt(v1+1e-8)
        lr2 = r0/np.sqrt(v2+1e-8)
        
        m1 += g1
        m2 += g2
    
        count += 1
        if count == nt_b:
            # Weight update
            #dW1 = lr1*m1*dt
            #dW2 = lr2*m2*dt
            dW1 = np.clip(lr1*m1*dt, -1e-3, 1e-3)
            dW2 = np.clip(lr2*m2*dt, -1e-3, 1e-3)
            connect_1.W = np.clip(connect_1.W+dW1, -0.1, 0.1)
            connect_2.W = np.clip(connect_2.W+dW2, -0.1, 0.1)
            
            m1 = np.zeros((N_mid, N_in))
            m2 = np.zeros((N_out, N_mid))
            v1 = np.zeros((N_mid, N_in))
            v2 = np.zeros((N_out, N_mid))

            count = 0
            
        # Save
        if i == num_iter-1:
            spike_arr[:,t] = s2
            c_arr[:, t] = h2
            v_arr[:, t] = neurons_2.v_
            e_arr[:, t] = e2[:, 0]
    
    cost_arr[i] = cost
    print("\n")
    print("cost:", cost)

# Plot
t = np.arange(nt)*dt
plt.figure(figsize=(6, 3))
for i in range(N_out):
    plt.plot(t, (v_arr[i]+60)/100 + i)
plt.title('LIF neuron')
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()
plt.show()

"""
plt.figure(figsize=(6, 3))
for i in range(N_out):
    plt.plot(t, (e_arr[i])/1.5 + i)
plt.title('Error')
plt.xlabel('Time (s)')
plt.ylabel('Membrane potential (mV)') 
plt.tight_layout()
plt.show()
"""
for i in range(N_out):
    plt.figure(figsize=(6, 3))
    plt.plot(t, (e_arr[i]))
    plt.title('Error'+str(i))
    plt.xlabel('Time (s)')
    plt.ylabel('Membrane potential (mV)') 
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(6, 3))
plt.plot(cost_arr)
plt.title('Cost')
plt.xlabel('Iter')
plt.ylabel('Cost') 
plt.tight_layout()
plt.show()