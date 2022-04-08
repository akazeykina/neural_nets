#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 14:03:52 2022

@author: ren
"""


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time



n_initial = 5000


def relu( x ):
    x = np.maximum(-5,x)
    return np.minimum( 5, x )
    #return np.maximum(0,x)

def trun(x) :
    x = np.maximum(-10,x)
    return np.minimum( 10, x )
    #return x


def initialize_parameters( d_z, d_y ):
    """
    Arguments:
    n_y -- the number of features in the data
    n_n -- the number of neurons (number of samples of nu)
    
    Returns:
    parameters -- python dictionary containing parameters "C", "W", "b"
                    C -- weight matrix of shape (n_n, 1, n_y)
                    W -- weight matrix of shape (n_n, n_y, n_y)
                    b -- bias vector of shape (n_n, n_y, 1)
    """
    
    norm = 20
    C = np.random.randn( n_initial, d_y , 1) * norm
    W = np.random.randn( n_initial, d_y , d_z) * norm
    b = np.random.randn( n_initial, d_y , 1)   * norm

        
    return ( C, W, b )


def F_potential(C, W, b, Z, Y, lam):
    
    (d_z, n_z) = Z.shape
    d_y = Y.shape[ 0 ]
    n_n = C.shape[ 0 ]
          
    AZ = relu( np.matmul( W, Z ) + b  ) * trun(C)    #dimension: n_n, d_y, n_z
    diff  = np.mean(AZ  , axis=0) - Y                      #dimension: d_y, n_z
    
    
    norm_square = np.squeeze(np.sum(C*C, axis =1)) + np.squeeze(np.sum(b*b, axis =1)) + np.squeeze(np.sum(W*W, axis =(1,2)))  
    
    error = np.mean(np.sum(diff*diff, axis=0)) + lam * np.mean(norm_square)
    
    dFdm = np.mean( np.sum(diff * AZ, axis = 1) , axis = 1) + lam * norm_square
    
    return dFdm, error

def Forward_propagation( C, W, b, Z, Y, dt, lam, sigma ):
    
    (d_z, n_z) = Z.shape
    d_y = Y.shape[ 0 ]
    n_n = C.shape[ 0 ]
    
    I = np.random.rand( n_n )
    
    dFdm, error = F_potential(C, W, b, Z, Y, lam)
    
    survive_index = (I > (dt * dFdm))
    
    C = C[survive_index, :, :] 
    W = W[survive_index, :, :] 
    b = b[survive_index, :, :] 
    
    n_n_new = C.shape[0]

    if n_n_new < n_initial/2:
        C = np.append(C, C, axis =0)
        W = np.append(W, W, axis =0)
        b = np.append(b, b, axis =0)
        n_n_new = 2 * n_n_new
    
    C += sigma * np.sqrt(dt) * np.random.randn( n_n_new, d_y, 1 )  
    W += sigma * np.sqrt(dt) * np.random.randn( n_n_new, d_y, d_z )  
    b += sigma * np.sqrt(dt) * np.random.randn( n_n_new, d_y, 1 )  
    
    
    return C, W, b, error, n_n_new
    
def my_onelayer_nn(Z, Y, dt = 0.1, lam = 0.01, sigma = 0.01, 
                  num_epochs = 1000):
    
    tic = time.time()
    #np.random.seed(1)
    
    errs = []
    n_particle = []

    #seed = 1 
    
    ( d_z , n_z ) = Z.shape
    
    d_y = Y.shape[0]
    
    
    #seed = 1 # for constructing minibatches
    
    # Parameters initialization.
    (C, W, b) = initialize_parameters( d_z, d_y )
   
    for i in range(0,num_epochs):
        C, W, b, error, n_n_new  = Forward_propagation( C, W, b, Z, Y, dt, lam, sigma )
        errs.append(error)
        n_particle.append(n_n_new)
            
    print(time.time() - tic)
   
    return C,W,b, errs, n_particle

Z =  np.linspace( 0 , 1, 50 )
Z =  Z.reshape(1, 50)               
#Y =  np.where((Z>0.3) & (Z<0.6),1,0)
Y = np.sin(5*2*np.pi*Z)/10



(C, W, b, errs, n_particle) = my_onelayer_nn( Z, Y, dt =0.001, lam = 0.0001, sigma = 0.4, num_epochs=10000)

LZ = np.matmul( W, Z ) + b       #dimension: n_n, d_y, n_z
AZ = relu( LZ )  
RZ = np.mean(AZ * trun(C) , axis=0)

fig = plt.figure( figsize = (8,3) )
ax0 = fig.add_subplot( 1, 2, 1 )
ax0.plot(  np.squeeze(RZ)  ) 
ax0.plot(np.squeeze(Y))

fig = plt.figure( figsize = (8,3) )
ax1 = fig.add_subplot( 1, 2, 1 )
ax1.plot(np.squeeze(errs[100:100000]), 'red')

fig = plt.figure( figsize = (8,3) )
ax2 = fig.add_subplot( 1, 2, 1 )
ax2.plot(np.squeeze(n_particle), 'blue')
    
    
    