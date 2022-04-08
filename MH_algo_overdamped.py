#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import wasserstein_distance


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def relu( x ):
    x = np.maximum(-10,x)
    return np.minimum( 10, x )


# In[3]:


def drelu( x ):
    return np.where( (x > -10) & (x < 10) , 1, 0 )


def trunC( x ):
    x = np.maximum(-100,x)
    return np.minimum(100, x)

# In[4]:


def my_Metropolis_Gaussian(p, dp, z0, sigma, d, n_samples=100, burn_in=0, m=1):
    """
    Metropolis Algorithm using a Gaussian proposal distribution.
    p: distribution that we want to sample from (can be unnormalized)
    dp: the derivative of p
    z0: Initial sample
    sigma: standard deviation of the proposal normal distribution.
    d: the dimension of the vector to simulate
    n_samples: number of final samples that we want to obtain.
    burn_in: number of initial samples to discard.
    m: this number is used to take every mth sample at the end
    """
    samples = np.zeros( ( d, n_samples ) )

    # Store initial value
    samples[ :, 0 ] = np.squeeze( z0 )
    z = z0
    # Compute the current likelihood
    l_cur = p( z )

    # Counter
    it = 0
    

    # Sample outside the for loop
    nsample_trial = 1000
    innov = np.random.normal(loc=0, scale=4, size=( d, nsample_trial + burn_in ) )
    u = np.random.rand( nsample_trial + burn_in )
    subsample = np.zeros( ( d, 1 ) )
    Fisher = 0

    while it < (nsample_trial + burn_in):
        # Random walk innovation on z
        cand = z + innov[ :, it ].reshape( ( d, 1 ) )

        # Compute candidate likelihood
        l_cand = p( cand )

        # Accept or reject candidate
        if l_cand - l_cur > np.log(u[ it ]):
            z = cand
            l_cur = l_cand

        #subsample = np.squeeze( z )
        if it > burn_in:
            subsample = z
            dp_vector = dp( subsample )
            Fisher += np.dot( dp_vector.T, dp_vector )
        
        it = it + 1
     
    Fisher = Fisher / nsample_trial
    sc_optimal = np.sqrt( 2.38 * 2.38 / Fisher / d )
    #sc_optimal = 5
    
    
    
    it = 0 
    # Total number of iterations to make to achieve desired number of samples
    iters = ( n_samples * m ) + burn_in
    innov = np.random.normal(loc=0, scale=sc_optimal, size=( d, iters ) )
    u = np.random.rand( iters )
    z = z0
    
    while it < iters:
        # Random walk innovation on z
        cand = z + innov[ :, it ].reshape( ( d, 1 ) )

        # Compute candidate likelihood
        l_cand = p( cand )

        # Accept or reject candidate
        if l_cand - l_cur > np.log(u[ it ]):
            z = cand
            l_cur = l_cand

        # Only keep iterations after burn-in and for every m-th iteration
        if it > burn_in and it % m == 0:
            samples[ :, ( it - burn_in ) // m ] = np.squeeze( z )

        it = it + 1

    return samples


# In[5]:


def p( x, C, W, b, sigma, lam_one ):
    
    phi = trunC(C) * relu( np.matmul( W, x ) + b )
    
    #theta_sq_norm = C**2 + np.linalg.norm( W, 
    #                axis = 1, keepdims = True )**2 + b**2
    
    x_sq_norm = x**2
    
    #result = np.exp( - 2 / ( sigma * sigma ) * ( np.mean( phi - lam * theta_sq_norm, axis = 0 )
    #                     + lam_one * x_sq_norm ) )
    #result = np.exp( - 2 / ( sigma * sigma ) * ( np.mean( phi , axis = 0 )
    #                     + lam_one * x_sq_norm ) )
    
    result =  - 2 / ( sigma * sigma ) * ( np.mean( phi , axis = 0 )
                         + lam_one * x_sq_norm ) 
    
    result = np.squeeze( result )
    
    return result


# In[6]:


def dp( x, C, W, b, sigma, lam_one ):
    
    dphi = trunC(C) * drelu( np.matmul( W, x ) + b ) * W
    
    result = - 2 / ( sigma * sigma ) * ( ( np.mean( dphi, axis = 0, keepdims = True ) ).T + 2 * lam_one * x )
    
    return result


# In[7]:


def initialize_parameters( n_y, n_n ):
    """
    Arguments:
    n_y -- the number of features in the data
    n_n -- the number of neurons (number of samples of nu)
    
    Returns:
    parameters -- python dictionary containing parameters "C", "W", "b"
                    C -- weight scalars of shape (n_n, 1)
                    W -- weight vector of shape (n_n, n_y)
                    b -- bias scalar of shape (n_n, 1)
    """
    
    parameters = {}

    parameters[ 'C' ] = np.random.randn( n_n, 1 ) * 12
    parameters[ 'W' ] = np.random.randn( n_n, n_y ) * 12
    parameters[ 'b' ] = np.random.randn( n_n, 1 ) * 12

        
    return parameters


# In[8]:


def forward_propagation( parameters, X, Z, dt, lam_one, lam, eps, sigma ):
    """
    Implement forward propagation (see formulas for dX, dTheta)
    
    Arguments:
    parameters -- current value of Theta = ( C, W, b )
    X -- samples of the current distribution mu, array of size ( n_y, n_x )
    z -- samples of the distribution mu_star, array of size ( n_y, n_z )
    dt, lam, eps, sigma -- parameters of gradient update rule
    
    Returns:
    X -- updated value of X
    parameters -- updated value of parameter Theta
    """
    
    C = parameters[ 'C' ]
    W = parameters[ 'W' ]
    b = parameters[ 'b' ]
    
    n_y, n_x = X.shape
    n_z = Z.shape[ 1 ]
    n_n = C.shape[ 0 ]
    
    # Update X using metropolis-hastings
    X = my_Metropolis_Gaussian( lambda x: p( x, C, W, b, sigma, lam_one ),
                                lambda x: dp( x, C, W, b, sigma, lam_one ),
                               z0 = np.zeros( ( n_y, 1 ) ), sigma = 0.1, d = n_y, n_samples = n_x, burn_in = 1000, m = 1 )
    #print( X.shape )
    
    LX = np.matmul( W, X ) + b
    AX = relu( LX )
    dAX = drelu( LX )
    BX = trunC(C) * dAX
    
    LZ = np.matmul( W, Z ) + b
    AZ = relu( LZ )
    dAZ = drelu( LZ )
    BZ = trunC(C) * dAZ 
    
    # Implement dTheta formula
    dc_dF_dnu = np.mean( AX, axis = 1, keepdims = True ) - np.mean( AZ, axis = 1, keepdims = True )
    dC = dc_dF_dnu * dt - 2 * lam * trunC(C) * dt + sigma * np.random.randn( n_n, 1 ) * np.sqrt( dt )
    parameters[ 'C' ] = C + dC
    
    dw_dF_dnu = np.matmul( BX, np.transpose( X, 
                                            ( 1, 0 ) ) ) / n_x - np.matmul( BZ, np.transpose( Z, ( 1, 0 ) ) ) / n_z 
    dW = dw_dF_dnu * dt - 2 * lam * W * dt + sigma * np.random.randn( n_n, n_y ) * np.sqrt( dt )
    parameters[ 'W' ] = W + dW
    
    db_dF_dnu = np.mean( BX, axis = 1, keepdims = True ) - np.mean( BZ, axis = 1, keepdims = True )
    db = db_dF_dnu * dt - 2 * lam * b * dt + sigma * np.random.randn( n_n, 1 ) * np.sqrt( dt )
    parameters[ 'b' ] = b + db
    
    return ( X, parameters )


# In[9]:


def compute_cost( parameters, X, Z ):
    """
    Implement the cost function defined by F.

    Arguments:
    parameters -- the parameters theta = ( C, W, b )
    X -- samples of the distribution mu, shape (n_y, n_x)
    Z -- samples of the distribution mu_star, shape (n_y, n_z)

    Returns:
    cost 
    """
    C = parameters[ 'C' ]
    W = parameters[ 'W' ]
    b = parameters[ 'b' ]
    
    cost = np.mean( - np.mean( trunC(C) * relu( np.matmul( W, X ) + b ), axis = 1, keepdims = True ) + 
                   np.mean( trunC(C) * relu( np.matmul( W, Z ) + b ), axis = 1, keepdims = True ), 
                   axis = 0, keepdims = True )
    
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return cost


# In[10]:


def nn_model(Z, n_n, n_x, minibatch_size = 32, dt = 0.05, lam_one = 0.01, lam = 0.1, eps = 1, sigma = 0.1, 
                  num_epochs = 1000, print_cost=True ):
    """
    Implements a 1-layer neural network.
    
    Arguments:
    Z -- data, samples of the distribution mu_star, numpy array of shape (n_y, n_z)
    n_n -- number of neurons in each layer (number of samples of distribution nu)
    n_x -- number of samples of the distribution mu
    minibatch_size -- size of minibatch
    dt -- parameter of the gradient update rule
    lam -- parameter of the gradient update rule
    eps -- parameter of the gradient update rule
    sigma -- parameters of the gradient update rule
    num_epochs -- number of epochs of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    costs -- list of costs F calculated at each iteration of the optimization loop
    X -- samples of the learned distribution mu, array of size ( n_y, n_x ) 
    """

    #np.random.seed(1)
    costs = []   # keep track of cost
    wdis = []
    cum_cost =0
    cum_w =0
    
    ( n_y, n_z ) = Z.shape
    
    #seed = 1 # for constructing minibatches
    
    # Parameters initialization.
    parameters = initialize_parameters( n_y, n_n )
    X = np.random.randn( n_y, n_x )

    # Loop (optimization)
    for i in range(0, num_epochs):
          
                
        # Forward propagation.
        X, parameters  = forward_propagation( parameters, X, Z, dt, lam_one, lam, eps, sigma )
        
        # Compute cost.
        cost = compute_cost( parameters, X, Z )
        cum_cost += np.abs(cost)
        w1 = wasserstein_distance(X.flatten(), Z.flatten())
        cum_w += w1
        
        if np.mod(i+1, 10)==0:
            wdis.append(cum_w/10)
            costs.append(cum_cost/10)
            
            if cum_cost/10 < 0.02:
                break
            
            cum_cost = 0
            cum_w = 0
            
            
        if print_cost:
            print ("Cost after epoch %i: %f" %(i, cost))

    
    return ( parameters, costs, wdis, X )


# In[11]:


# Test on a simple distribution
a =  np.random.randn( 1, 1000 ) - 1
b = np.random.randn(  1, 1000 ) + 4
Z= np.concatenate((b,a), axis=1)

#Z = 2 * np.random.rand( 1, 1000 )
parameters, costs, wdis, X = nn_model( Z, n_n = 3000, n_x =2000, dt=0.0001, lam_one = 0.02, lam = 0.02, 
                                sigma = 1, num_epochs = 500)


# In[12]:


# In[13]:


plt.figure( figsize = (5,3) )
plt.hist( X.T, bins=50, alpha=0.5, label = 'GAN' )
plt.hist(Z.T, bins=50, alpha=0.5, label = 'data' )
plt.legend()
plt.title('Comparison of Histograms')

# In[ ]:

plt.figure( figsize = (5,3) )
plt.plot( costs,'red', label = 'potential' )
plt.plot( wdis,'g--', label = 'W1-distance' )
plt.xlabel('iteration * 10')
plt.ylabel('energy')
plt.legend()
plt.title('Training error - Overdamped')



# In[ ]:




