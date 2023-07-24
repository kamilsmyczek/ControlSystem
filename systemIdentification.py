# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:20:07 2023

@author: kamil
"""
from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
from scipy.interpolate import CubicSpline


FILENAME = 'systemIdentData.csv'
EPS = 1e-9
df = pd.read_csv(FILENAME)

Obj_t = df['t']
Obj_u = df['control']
Obj_y = df['water']

Obj_dt = Obj_t[1:].reset_index(drop=True).subtract(Obj_t[:-1]).mean()
print('Object sample time:', Obj_dt)


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

T_end = Obj_t.max(0)
dt = 1
t = np.arange(T_end + dt, step=dt)
u_t = Obj_u#createControlVector(t, Obj_t, Obj_u)
for i in range(len(Obj_t)):
    if Obj_u[i] != u_t[i]:
        print('U diff')

u_t2 = list(zip(u_t, [23] * len(u_t)))
x0 = [20]

def interpolateData(t, v):
    x, y = zip(*v)
    f_t = scipy.interpolate.CubicSpline(x, y)
    return [f_t(_t) for _t in t]

def createSystem(a):
    # System matrices
    A = [[-a[1], a[1]], [a[2], -a[2] - a[3]]]
    B = [[a[0], 0], [0, a[3]]]
    C = [[0, 1]]
    D = 0
      
    return ss(A, B, C, D)

x0 = [[23], [23]]
def Quality(alpha):
    yout, T, xout = lsim(createSystem(alpha), U=u_t2, T=t, X0=x0)
    Sim_y = interpolateData(Obj_t, zip(T, yout))

    q = 0
    for v in zip(Sim_y, Obj_y):
        q += abs(v[0]- v[1])
    # add penalty for negative coefs
    for a in alpha:
        if a < 0:
            q += (-a) * 100000
            
    print('Q(', alpha, ') = ', q, sep='')
    return q

#%%
plt.close()
plt.figure(1)
plt.plot(Obj_t, Obj_y, Obj_t, Obj_u)
plt.legend(['Y', 'U'])
plt.grid()
plt.show()  
a0 = [0.14, 0.025, 0.03, 0.016] #[0.1, 0.045, 0.01, 0.007] #[0.14, 0.025, 0.03, 0.016]
#print('Quality(a0)=', Quality(a0))
optAlpha = scipy.optimize.fmin(func=Quality, x0=a0)
print('Optimum value:', optAlpha)
#%
optSys = createSystem(optAlpha)
yout, T, xout = lsim(optSys, U=u_t2, T=t, X0=x0)
plt.figure(2)
plt.plot(T, yout, Obj_t, Obj_y, '.')
plt.legend(['Sim best', 'Object'])
plt.grid()
plt.show()