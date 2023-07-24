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
from control.matlab import *
import numpy as np


alpha = [0.1, 0.045, 0.01, 0.007]

t_end = 700

ADD_NOISE = False

T_ambient = 23
T_heater0 = [T_ambient]
T_water0 = [T_ambient]

dt = 1
control = [(0, 0), (200, 10), (300, 0)]

random.seed(time.time())


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


def getNoise(maxVal, minVal):
    if not ADD_NOISE:
        return 0
    
    r = random.random()
    return r * (maxVal - minVal) + minVal


def createSystem(a):
    # System matrices
    A = [[-a[1], a[1]], [a[2], -a[2] - a[3]]]
    B = [[a[0], 0], [0, a[3]]]
    C = [[0, 1]]
    D = 0
      
    return ss(A, B, C, D)

t = np.arange(0, t_end, dt)
u_t = createControlVector(t, list(zip(*control))[0], list(zip(*control))[1])
u2_t = list(zip(u_t, [T_ambient] * len(u_t)))
x0 = [T_heater0, T_water0]
yout, T, xout = lsim(createSystem(alpha), U=u2_t, T=t, X0=x0)
            
dataset = {'t': t, 'control': u_t, 'water': next(zip(*yout))}

plt.close('all')
plt.figure(1)
plt.plot(t, u_t, '.', T, yout, '.')
plt.legend(['u', 'y'])
plt.grid()
plt.show()

df = pd.DataFrame(dataset)
# saving the dataframe
df.to_csv('systemIdentData.csv')

