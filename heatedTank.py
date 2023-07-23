# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 21:50:49 2023

@author: kamil
"""
import numpy as np
import matplotlib.pyplot as plt

T_ambient = 20
T_heater = T_ambient
T_water = 1.3* T_ambient
T_heater0 = T_heater
T_water0 = T_water
dt = 0.2
T_end = 1200
power = 10
alpha = [0.1, 0.045]
beta = [0.01, 0.007]
manual = True
control = [(0, 0), (100, 10), (300, 0), (800, 30), (999999, 0)]
setpoint = [(0, 30), (200, 120), (300, 30), (999999, 30)]
K_pid = [0.001, 0.05, 0.1]

def nextHeaterTemp(t, p):
    global T_heater
    dT = p * alpha[0] - (T_heater - T_water) *  alpha[1] 
    T_heater += dT * dt
    return T_heater


def nextWaterTemp(t):
    global T_water
    dT = (T_heater - T_water) * beta[0] - (T_water - T_ambient) *  beta[1] 
    T_water += dT * dt
    return T_water

class PID:
    def __init__(self, _K, _dt, _maxOut=[0, 100]):
        self.K = _K
        self.dt = _dt
        self.Iterm = 0
        self.lastErr = 0
        self.maxOut = _maxOut[:]
        self.maxOut.sort()
    
    def compute(self, ref, act):
        err = ref - act

        self.Iterm += err * self.K[1] * self.dt;
        if self.Iterm > self.maxOut[1]:
            self.Iterm = self.maxOut[1]
            #print('Przekroczony maks')
        elif self.Iterm < self.maxOut[0]:
            self.Iterm = self.maxOut[0]
            #print('Przekroczony min')
            
        errDiff = (err - self.lastErr) / self.dt
        
        out = err * self.K[0] + self.Iterm + errDiff * self.K[2]
        if out > self.maxOut[1]:
            out = self.maxOut[1]
        elif out < self.maxOut[0]:
            out = self.maxOut[0]
            
        return out
        
        

t = np.arange(T_end, step=dt)
simHeater = []
simWater = []
idxCtrl = 1
ctrl = control[0][1]
setTemp = setpoint[0][1]

def createControlVector(_t, _t_act, _y_act):
    # Check asscending order of time in control vector
    t_prev = _t_act[0]
    for i in _t_act[1:]:
        assert i > t_prev, 'Time in proposed control vector not in ascending order'
        t_prev = i
        
    u_t = []
    u_idx = 1
    u_now = _y_act[0]
    for t in _t:
        if u_idx < len(_y_act) and t > _t_act[u_idx]:
            u_now = _y_act[u_idx]
            u_idx += 1
        u_t.append(u_now)
    
    return u_t
t_ctrl, u_ctrl = zip(*control)
u_t = createControlVector(t, t_ctrl, u_ctrl)
setVector = []
controlVector = []
regulator = PID(K_pid, dt)
for n, i in enumerate(t):
    if manual:        
        ctrl = u_t[n]
        idxCtrl += 1
    else:
        if i > setpoint[idxCtrl][0]:
            setTemp = setpoint[idxCtrl][1]
            idxCtrl += 1
        ctrl = regulator.compute(setTemp, T_water)
    simHeater.append(nextHeaterTemp(i, ctrl))    
    simWater.append(nextWaterTemp(i))
    controlVector.append(ctrl)
    setVector.append(setTemp)
    print('t:', i)
    
    
###############################################################################


A = [[-alpha[1], alpha[1]], [beta[0], - beta[0] - beta[1]]]
B = [[alpha[0], 0], [0, beta[1]]]
C = [[0, 1]]
D = 0
from control.matlab import *
sys = ss(A, B, C, D)
u_t2 = list(zip(u_t, [T_ambient] * len(u_t)))
# Step response for the system
x0 = [[T_heater0], [T_water0]]
print('x0:', x0)
yout, T, xout = lsim(sys, U=u_t2, T=t, X0=x0)

plt.close('all')
plt.plot(t, simHeater, t, simWater, t, controlVector, T[:-1], yout[1:], '.')
plt.legend(['heater', 'water', 'power', 'ctrl lib'])   
plt.xlabel('t [s]')
plt.ylabel('Temperature [degC]')

plt.grid()
plt.show()


EXIT
plt.close('all')
if manual:
    plt.plot(t, simHeater, t, simWater, t, controlVector)
    plt.legend(['heater', 'water', 'power'])
else:
    plt.plot(t, simHeater, t, simWater, t, controlVector, t, setVector)
    plt.legend(['heater', 'water', 'power', 'setpoint'])    
plt.xlabel('t [s]')
plt.ylabel('Temperature [degC]')

plt.grid()
plt.show()