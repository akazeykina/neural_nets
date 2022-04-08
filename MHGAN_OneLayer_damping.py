
# coding: utf-8

# In[707]:


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import wasserstein_distance

get_ipython().run_line_magic('matplotlib', 'inline')


# In[708]:


def relu( x ):
    x = np.maximum(-10,x)
    return np.minimum( 10, x )
#    return np.maximum( 0, x )


# In[709]:


def drelu( x ):
    return np.where( (x > -10) & (x < 10) , 1, 0 )
#    return np.where( x > 0, 1, 0 )


# In[710]:


def trunC( x ):
    x = np.maximum(-100,x)
    return np.minimum(100, x)


# In[711]:


def dtrunC(x):
    return np.where( (x > -100) & (x < 100) , 1, 0 )


# In[712]:


def my_Metropolis_Gaussian(p, dp, z0, sigma, d, n_samples=1000, burn_in=0, m=1):
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
    #sc_optimal = 4
    
    
    
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


# In[713]:


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


# In[714]:


def dp( x, C, W, b, sigma, lam_one ):
    
    dphi = trunC(C) * drelu( np.matmul( W, x ) + b ) * W
    
    result = - 2 / ( sigma * sigma ) * ( ( np.mean( dphi, axis = 0, keepdims = True ) ).T + 2 * lam_one * x )
    
    return result


# In[715]:


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

    parameters[ 'C' ] = np.random.randn( n_n, 1 )*12
    parameters[ 'Cv'] = np.random.randn( n_n, 1 ) *  0.2
    
    parameters[ 'W' ] = np.random.randn( n_n, n_y )*12
    parameters[ 'Wv' ] = np.random.randn( n_n, n_y ) * 0.2
    
    parameters[ 'b' ] = np.random.randn( n_n, 1 )*12
    parameters[ 'bv' ] = np.random.randn( n_n, 1 ) * 0.2

        
    return parameters


# In[716]:
def drift_derv(C, W, b, X, Z, n_x, n_z):
    
    LX = np.matmul( W, X ) + b
    AX = relu( LX )
    dAX = drelu( LX )
    BX = trunC(C) * dAX
    
    LZ = np.matmul( W, Z ) + b
    AZ = relu( LZ )
    dAZ = drelu( LZ )
    BZ = trunC(C) * dAZ 
    
    # Implement dTheta formula
    dc_dF_dnu = ( np.mean( AX, axis = 1, keepdims = True ) - np.mean( AZ, axis = 1, keepdims = True ) ) * dtrunC(C)    
    dw_dF_dnu = np.matmul( BX, np.transpose( X, 
                                            ( 1, 0 ) ) ) / n_x - np.matmul( BZ, np.transpose( Z, ( 1, 0 ) ) ) / n_z
    db_dF_dnu = np.mean( BX, axis = 1, keepdims = True ) - np.mean( BZ, axis = 1, keepdims = True )

    return dc_dF_dnu, dw_dF_dnu, db_dF_dnu


    

def forward_propagation( parameters, X, Z, dt, lam_one, eta, lam, gamma, sigma , dc_dF_dnu, dw_dF_dnu, db_dF_dnu):
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
    Cv = parameters[ 'Cv' ]
    
    W = parameters[ 'W' ]
    Wv = parameters[ 'Wv' ]
    
    b = parameters[ 'b' ]
    bv = parameters[ 'bv' ]
    
    n_y, n_x = X.shape
    n_z = Z.shape[ 1 ]
    n_n = C.shape[ 0 ]
    
    # Update X using metropolis-hastings
    X = my_Metropolis_Gaussian( lambda x: p( x, C, W, b, sigma, lam_one ),
                                lambda x: dp( x, C, W, b, sigma, lam_one ),
                               z0 = np.zeros( ( n_y, 1 ) ), sigma = 0.1, d = n_y, n_samples = n_x, burn_in = 1500, m = 1 )
    #print( X.shape )
    
    vol = sigma * np.sqrt((1-np.exp(-dt*gamma))/2/gamma)
    
    Cv = Cv * np.exp(-dt*gamma/2) + vol  * np.random.randn( n_n, 1 )
    Cv += dc_dF_dnu * dt/2 - lam * trunC(C) * dt/2
    C  += eta * Cv * dt
    parameters[ 'C' ] = C 
    
    Wv = Wv * np.exp(-dt*gamma/2) + vol  * np.random.randn( n_n, n_y )
    Wv += dw_dF_dnu * dt/2 - lam * W * dt/2
    W  += eta * Wv * dt
    parameters[ 'W' ] = W 
    
    bv = bv * np.exp(-dt*gamma/2) + vol  * np.random.randn( n_n, 1 )
    bv += db_dF_dnu * dt/2 - lam * b * dt/2
    b  += eta * bv * dt
    parameters[ 'b' ] = b
    
    dc_dF_dnu, dw_dF_dnu, db_dF_dnu = drift_derv( C, W, b, X, Z, n_x, n_z ) 
    
    Cv += dc_dF_dnu * dt/2 - lam * trunC(C) * dt/2
    parameters[ 'Cv' ] = Cv * np.exp(-dt*gamma/2) + vol * np.random.randn( n_n, 1 )

    Wv += dw_dF_dnu * dt/2 - lam * W * dt/2
    parameters[ 'Wv' ] = Wv * np.exp(-dt*gamma/2) + vol * np.random.randn( n_n, n_y )
    
    bv += db_dF_dnu * dt/2 - lam * b * dt/2
    parameters[ 'bv' ] = bv * np.exp(-dt*gamma/2) + vol * np.random.randn( n_n, 1 )
    
   
    
    # dc_dF_dnu, dw_dF_dnu, db_dF_dnu = drift_derv( C, W, b, X, Z, n_x, n_z )    
    
    # Cv += dc_dF_dnu * dt/2 - (lam * C + gamma * Cv) * dt + sigma * np.random.randn( n_n, 1 ) * np.sqrt( dt/2 )
    # C  += eta* Cv * dt
    # Wv += dw_dF_dnu * dt/2 - (lam * W + gamma * Wv) * dt + sigma * np.random.randn( n_n, n_y ) * np.sqrt( dt/2 )
    # W  += eta* Wv * dt
    # bv += db_dF_dnu * dt/2 - (lam * b + gamma * bv) * dt + sigma * np.random.randn( n_n, 1 ) * np.sqrt( dt/2 )
    # b  += eta* bv * dt
    
    # dc_dF_dnu, dw_dF_dnu, db_dF_dnu = drift_derv( C, W, b, X, Z, n_x, n_z )
    
    # parameters[ 'Cv' ] = (Cv + dc_dF_dnu * dt/2 - lam * C * dt + sigma * np.random.randn( n_n, 1 ) * np.sqrt( dt/2 ) ) / (1 + gamma * dt/2)
    # parameters[ 'C' ]  = C
    
    
    # parameters[ 'Wv' ] = (Wv + dw_dF_dnu * dt/2 - lam * W * dt + sigma * np.random.randn( n_n, n_y ) * np.sqrt( dt/2 ) ) / (1 + gamma * dt/2)
    # parameters[ 'W' ]  = W 
    
    
    # parameters[ 'bv' ] = (bv + db_dF_dnu * dt/2 - lam * b * dt + sigma * np.random.randn( n_n, 1 ) * np.sqrt( dt/2 )) / (1 + gamma * dt/2)
    # parameters[ 'b' ]  = b 
    
    return ( X, parameters, dc_dF_dnu, dw_dF_dnu, db_dF_dnu )


# In[717]:


def compute_cost( parameters, X, Z ,eta):
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
    Cv = parameters[ 'Cv' ]
    
    W = parameters[ 'W' ]
    Wv = parameters[ 'Wv' ]
    
    b = parameters[ 'b' ]
    bv = parameters[ 'bv' ]
    
    
    n_n = C.shape[0]
    
    potential = np.mean( - np.mean( trunC(C) * relu( np.matmul( W, X ) + b ), axis = 1) + 
                   np.mean( trunC(C) * relu( np.matmul( W, Z ) + b ), axis = 1),  axis = 0 )
    
    cost = potential + 0.5* eta * np.sum(Cv*Cv + Wv*Wv + bv*bv)/n_n
    
    
    potential = np.squeeze(potential)
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    
    
    return (cost, potential)


# In[718]:


def nn_model(Z, n_n, n_x, minibatch_size = 32, dt = 0.05, lam_one = 0.01, eta = 0.2, lam = 0.1, gamma = 0.5, sigma = 0.4, 
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
    pots =[]
    wdis =[]
    
    ( n_y, n_z ) = Z.shape
    
    #seed = 1 # for constructing minibatches
    
    # Parameters initialization.
    parameters = initialize_parameters( n_y, n_n )
    X = np.random.randn( n_y, n_x )

    cum_cost = 0
    cum_po = 0
    cum_w = 0
    # Loop (optimization)
    
    
    C = parameters[ 'C' ]
    
    W = parameters[ 'W' ]
    
    b = parameters[ 'b' ]
    
    n_y, n_x = X.shape
    n_z = Z.shape[ 1 ]
    n_n = C.shape[ 0 ]
    
    
    dc_dF_dnu, dw_dF_dnu, db_dF_dnu = drift_derv( C, W, b, X, Z, n_x, n_z ) 
    
    

    
    for i in range(0, num_epochs):
          
                
        # Forward propagation.
        (X, parameters, dc_dF_dnu, dw_dF_dnu, db_dF_dnu)  = forward_propagation( parameters, X, Z, dt, lam_one, eta, lam, gamma, sigma , dc_dF_dnu, dw_dF_dnu, db_dF_dnu)
        
        # Compute cost.
        (cost, po) = compute_cost( parameters, X, Z ,eta)
        
        cum_w += wasserstein_distance(X.flatten(), Z.flatten())
        cum_cost += cost
        cum_po += np.abs(po)
        
        if np.mod(i+1, 10)==0:
            wdis.append(cum_w/10)
            costs.append(cum_cost/10)
            pots.append(cum_po/10)
            
            if cum_po/10 < 0.02:
                break
            
            cum_cost = 0
            cum_po = 0
            cum_w = 0
            
            
        if print_cost:
            print ("Cost after epoch %i: %f" %(i, cost))
            print ("Potential after epoch %i: %f" %(i, po))

            
        

    
    return ( parameters, costs, pots, wdis, X )


# In[735]:


# Test on a simple distribution
#Z = np.random.exponential( 1, ( 1, 2000 ) )
a =  np.random.randn( 1, 1000 ) - 1
b = np.random.randn(  1, 1000 ) + 4
Z= np.concatenate((b,a), axis=1)
#Z = 2 * np.random.rand( 1, 1000 )
parameters, costs, pots, wdis, X = nn_model( Z, n_n = 3000, n_x = 2000, dt = 0.005, lam_one = 0.02, eta = 0.3, lam = 0.02, gamma = 2,
                                sigma = 1, num_epochs = 2000)


# In[736]:


plt.figure( figsize = (5,3) )
plt.plot(pots, 'red', label = 'potential')
plt.plot( costs,'blue', label = '+ kinetic' )
plt.plot( wdis,'g--', label = 'W1-distance' )
plt.xlabel('iteration * 10')
plt.ylabel('energy')
plt.legend()
plt.title('Training error')




# In[737]:


plt.figure( figsize = (5,3) )
plt.hist( X.T, bins=50, alpha=0.5, label = 'GAN' )
plt.hist(Z.T, bins=50, alpha=0.5, label = 'data' )
plt.legend()
plt.title('Comparison of Histograms')

