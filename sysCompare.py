# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:20:07 2023

@author: kamil
"""

from control.matlab import *
import matplotlib.pyplot as plt
import numpy as np


Resistance = 1 * 10**3;
Capacity = 10 * 10**-6
T_end = 0.7
U = [(0, 0), (0.1, 1), (0.5, 2), (0.6, 0)]
                                  
dt = 0.001

# System matrices
A = [-1/(Resistance * Capacity)]
B = [1 / (Resistance * Capacity)]
C = [1.]


sys = ss(A, B, C, 0)

# Step response for the system
plt.figure(1)
yout, T = step(sys, T_end)

Vout = [0]
Vin = 1


def createControlVector(_t, _u):
    # Check asscending order of time in control vector
    t_prev = _u[0][0]
    for i in _u[1:]:
        assert i[0] > t_prev, 'Time in proposed control vector not in ascending order'
        t_prev = i[0]
        
    u_t = []
    u_idx = 1
    u_now = _u[0][1]
    for t in _t:
        if u_idx < len(_u) and t > _u[u_idx][0]:
            u_now = _u[u_idx][1]
            u_idx += 1
        u_t.append(u_now)
    
    return u_t
        
    
    
    
def nextState(t, x, u, dt):
    dx = (u - x[-1]) / (Resistance * Capacity)
    return x[-1] + dx * dt


t = np.arange(T_end, step=dt)
u_t = createControlVector(t, U)


for i, t_now in enumerate(t):
    Vout.append(nextState(t_now, Vout, u_t[i], dt))
Vout = Vout[:-1]
yout, T, xout = lsim(sys, U=u_t, T=t, X0=0.0)

plt.close('all')
plt.figure(1)
plt.plot(T.T, yout.T, 'r', t, Vout, 'b.', t, u_t, 'g.')
plt.legend(['Ctr', 'my', 'u'])
plt.show(block=False)


import scipy.optimize
from scipy.interpolate import CubicSpline
def interpolateData(t, v):
    x, y = zip(*v)
    f_t = scipy.interpolate.CubicSpline(x, y)
    return [f_t(_t) for _t in t]


def Quality(a, b):
    q = 0
    for v in zip(a, b):
        q += abs(v[0]- v[1])
    q = 0
    for i in range(len(a)):
        q += abs(a[i] - b[i])
    return q
    
print('Quality:', Quality(yout, Vout)) 
banana = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
xopt = scipy.optimize.fmin(func=banana, x0=[-1.2,1])

plt.figure(2)
v = [1,2,3,4,5,6,7,8,9]
v2 = [2.5, 3.6,3.9]
plt.plot(v, v, v2, interpolateData(v2, zip(v,v)), '.')