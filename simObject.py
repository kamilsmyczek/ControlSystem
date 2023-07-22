# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 10:20:24 2023

@author: kamil
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd

alpha = [0.1, 0.045]
beta = [0.01, 0.007]

t_end = 600

ADD_NOISE = False

T_ambient = 20
T_heater = [T_ambient]
T_water = [1.3 * T_ambient]

dt = 0.2
control = [(0, 0), (200, 10), (300, 0), (999999, 0)]

random.seed(time.time())


def nextHeaterTemp(t, p):
    dT = p * alpha[0] - (T_heater[-1] - T_water[-1]) *  alpha[1] 
    T_next = T_heater[-1] + dT * dt
    return T_next


def nextWaterTemp(t):
    dT = (T_heater[-1] - T_water[-1]) * beta[0] - (T_water[-1] - T_ambient) *  beta[1] 
    T_next = T_water[-1] + dT * dt
    return T_next

def getNoise(maxVal, minVal):
    if not ADD_NOISE:
        return 0
    
    r = random.random()
    return r * (maxVal - minVal) + minVal

idxCtrl = 0
ctrl = control[idxCtrl][1]

time = np.arange(t_end, step=dt)
power = [ctrl]
timeNoise = []
for t in time:
    if t > control[idxCtrl][0]:
        ctrl = control[idxCtrl][1]
        idxCtrl += 1
    
    T_heater.append(nextHeaterTemp(t, ctrl) + getNoise(.1, -.1))
    T_water.append(nextWaterTemp(t) + getNoise(.1, -.1))
    power.append(ctrl)
    timeNoise.append(t + getNoise(dt * 0.4, -dt * 0.4))
size = len(time)
T_heater = T_heater[:size]
T_water = T_water[:size]
power = power[:size]  
plt.close('all')
#plt.plot(time, timeNoise, '.', time, time)
plt.plot(timeNoise, T_heater, timeNoise, T_water, timeNoise, power)
#plt.legend(['heater', 'water', 'power'])
plt.grid('minor') 
plt.xlabel('t [s]')
plt.ylabel('Temperature [degC]')

dataset = {'t': timeNoise, 'control': power,  'heater': T_heater, 'water': T_water}

df = pd.DataFrame(dataset)
     
# saving the dataframe
df.to_csv('GFG.csv')

#
from control.matlab import *
# System matrices
A = [[0, 1.], [-k/m, -b/m]]
B = [[0], [1/m]]
C = [[1., 0]]
sys = ss(A, B, C, 0)
