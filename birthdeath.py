#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:05:07 2021

@author: ren
"""
import numpy as np
import matplotlib.pyplot as plt


def F_potential(x):
    f1 = np.minimum(3*(x-1)*(x-1)+0.5, 5)
    f2 = np.minimum((x+10)*(x+10), 5)
    return np.minimum(f1, f2)

def birthdeath(x, sigma, dt):
    
    y = []
    n, m = x.shape 
    
    x = x + sigma * np.sqrt(dt) * np.random.randn( n, 1 )
    I = np.random.rand( n, 1 )
    for i in range(0,n):
        if I[i] > F_potential(x[i]) * dt:
            y.append(x[i])
    return y

def my_iteration(n, num_iteration, sigma,dt):
    
    x = np.random.randn( n, 1 )*12
    cost = []
            
    for i in range(0, num_iteration):
        x = np.array(birthdeath(x, sigma, dt))
        k0, k1 = x.shape
        if k0 < n/2:
            np.append(x, x)
        cost.append(np.log(np.mean(F_potential(x)))) 
        
    plt.figure()
    plt.hist(x)
    
    plt.figure()
    plt.plot(cost)


my_iteration(20000, 300, 0.1, 0.01)