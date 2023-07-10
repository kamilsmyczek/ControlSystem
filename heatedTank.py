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
dt = 0.2
T_end = 600
power = 10
alpha = [0.1, 0.045]
beta = [0.01, 0.007]
manual = False
control = [(0, 0), (200, 10), (300, 0), (999999, 0)]
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
        """if self.Iterm > self.maxOut[1]:
            self.Iterm = self.maxOut[1]
            print('PRzekracz maks')
        elif self.Iterm < self.maxOut[0]:
            self.Iterm = self.maxOut[0]
            #print('PRzekracz min')
        """
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
controlVector = []
setVector = []
regulator = PID(K_pid, dt)
for i in t:
    if manual:
        if i > control[idxCtrl][0]:
            ctrl = control[idxCtrl][1]
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