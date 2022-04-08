#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:57:54 2021

@author: ren
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time


def reg_stop(m, n, X0, alpha, eps, T=1):
    # m is number of particles
    # n is number of time grids
    
#    X = np.zeros((m,n))
#    X[:, 0] = X0
    X = X0
    intX = 0
    prob = np.zeros(n)
    prob[0] = alpha
    
    dt = T/n
  
    
    for i in range(1, n):
        
        #intX =+ 1/eps * np.minimum(X[:, i-1], 0) * dt
        intX = intX + 1/eps * np.minimum(X,0) * dt
        prob[i] = alpha * np.mean(np.exp(intX))
        
        X = X - np.sin(10*X)*dt + np.random.randn(m)*np.sqrt(dt) + prob[i] - prob[i-1]
       
    return np.squeeze(1-prob/alpha)

    

m = 40000
n = 2000
X0 = np.random.randn(m)*2+5
T = 5

appx = np.zeros((5, n))

for i in range(5):
    eps = 1/np.power(10, i)
    appx[i,:] = reg_stop(m, n, X0, 10, eps, T)
    plt.plot(np.linspace(0,T,n), appx[i,:], label = 'epsilon='+str(eps))
    
plt.legend()