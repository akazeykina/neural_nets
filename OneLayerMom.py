
# coding: utf-8

# In[14]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time


#get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


def relu( x ):
    x = np.maximum(0,x)
    return np.minimum( 10, x )
#    return np.maximum(0,x)

def trun(x) :
    x = np.maximum(-200,x)
    return np.minimum( 200, x )

# In[16]:


def drelu( x ):
    return np.where( (x > 0) & (x < 10) , 1, 0 )
#    return np.where(x>0, 1, 0)

def dtrun( x ):
    return np.where( (x > -200) & (x < 200) , 1, 0 )

# In[17]:


def initialize_parameters( d_z, d_y, n_n ):
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
    C = np.random.randn( n_n, d_y , 1) * norm
    W = np.random.randn( n_n, d_y , d_z) * norm
    b = np.random.randn( n_n, d_y , 1)   * norm

        
    return ( C, W, b )


# In[18]:


def forward_propagation( C, W, b, Z, Y, dt, lam, sigma):
    """
    Implement forward propagation (see formulas for dX, dTheta)
    
    Arguments:
    parameters -- current value of Theta = ( C, W, b )
    Z -- data, array of size ( d_z, n_z )
    Y -- label, array of size ( d_y, n_z)
    dt, lam, sigma -- parameters of gradient update rule
    
    Returns:

    parameters -- updated value of parameter Theta
    """
    
    (d_z, n_z) = Z.shape
    d_y = Y.shape[ 0 ]
    n_n = C.shape[ 0 ]
        
    
    LZ = np.matmul( W, Z ) + b       #dimension: n_n, d_y, n_z
    AZ = relu( LZ )  
    #RZ = np.mean(AZ * trun(C) , axis=0)
    RZ = AZ * trun(C) 
    
    
    diff = np.mean(RZ, axis=0) - Y

    
    dAZ = drelu( LZ )
    BZ = trun(C) * dAZ * diff
    
    # Implement dTheta formula
    dc_dF_dnu = np.mean( AZ* diff , axis = 2, keepdims = True ) * dtrun(C)        # dimension:  n_n, d_y, 1
    
    dw_dF_dnu = np.matmul(BZ , Z.T)/n_z               # dimension:  n_n, d_y, d_z
    db_dF_dnu = np.mean( BZ , axis = 2, keepdims = True )        # dimension:  n_n, d_y, 1
    
    dC = - ( dc_dF_dnu + lam * C ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, 1 ) 
    C += dC
    
    dW = - ( dw_dF_dnu + lam * W ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, d_z )
    W += dW
    
    db = - ( db_dF_dnu + lam * b ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, 1 )
    b += db
    
    
    
    cost = np.sum(diff * diff)/2/n_z
    
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return (cost, C, W, b)





def forward_propagation_MOM( C, W, b, Z, Y, dt, lam, sigma , K):
    """
    Implement forward propagation (see formulas for dX, dTheta)
    
    Arguments:
    parameters -- current value of Theta = ( C, W, b )
    Z -- data, array of size ( d_z, n_z )
    Y -- label, array of size ( d_y, n_z)
    dt, lam, sigma -- parameters of gradient update rule
    
    Returns:

    parameters -- updated value of parameter Theta
    """
    
    (d_z, n_z) = Z.shape
    d_y = Y.shape[ 0 ]
    n_n = C.shape[ 0 ]
        
    
    LZ = np.matmul( W, Z ) + b       #dimension: n_n, d_y, n_z
    AZ = relu( LZ )  
    #RZ = np.mean(AZ * trun(C) , axis=0)
    RZ = AZ * trun(C) 
    
    
    MO = np.zeros([K, d_y, n_z])
    subsize = int(np.floor(n_n/K))
    for i in range(0,int(K-1)):
        MO[i,:,:]= np.mean(RZ[ (i*subsize) : ((i+1)*subsize),:,:],axis=0)
    
    MO[K-1,:,:] = np.mean(RZ[((K-1)*subsize) : n_n, : , :], axis=0)
    MOM = np.median(MO, axis =0)
    
    diff =  MOM  - Y       # dimension:  d_y, n_z

    
    dAZ = drelu( LZ )
    BZ = trun(C) * dAZ * diff
    
    # Implement dTheta formula
    dc_dF_dnu = np.mean( AZ* diff , axis = 2, keepdims = True ) * dtrun(C)        # dimension:  n_n, d_y, 1
    
    dw_dF_dnu = np.matmul(BZ , Z.T)/n_z               # dimension:  n_n, d_y, d_z
    db_dF_dnu = np.mean( BZ , axis = 2, keepdims = True )        # dimension:  n_n, d_y, 1
    
    dC = - ( dc_dF_dnu + lam * C ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, 1 ) 
    C += dC
    
    dW = - ( dw_dF_dnu + lam * W ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, d_z )
    W += dW
    
    db = - ( db_dF_dnu + lam * b ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, 1 )
    b += db
    
    
    
    cost = np.sum(diff * diff)/2/n_z
    
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return (cost, C, W, b)





# In[19]:


def validation(ZV, YV, C, W, b):
    
    m = ZV.shape[1]
    
    LZ = np.matmul( W, ZV ) + b       #dimension: n_n, d_y, n_z
    AZ = relu( LZ )  
    Dif = np.mean(AZ * trun(C) , axis=0) - YV
    
    err = np.sum(Dif * Dif)/2/m
        

    return err

def validation_MOM(ZV, YV, C, W, b, K):
    (d_z, n_z) = ZV.shape
    d_y = YV.shape[ 0 ]
    n_n = C.shape[ 0 ]
    
    LZ = np.matmul( W, ZV ) + b       #dimension: n_n, d_y, n_z
    AZ = relu( LZ ) * trun(C)
    
    MO = np.zeros([K, d_y, n_z])
    subsize = int(np.floor(n_n/K))
    for i in range(0,int(K-1)):
        MO[i,:,:]= np.mean(AZ[ (i*subsize) : ((i+1)*subsize),:,:],axis=0)
    
    MO[K-1,:,:] = np.mean(AZ[((K-1)*subsize) : n_n, : , :], axis=0)
    MOM = np.median(MO, axis =0)
    
    Dif =  MOM  - YV  
    err = np.sum(Dif * Dif)/2/n_z
        

    return err



def nn_model(Z, Y, ZV, YV, n_n, minibatch_size = 32, dt = 0.1, lam = 0.01, sigma = 0.01, K = 1,
                  num_epochs = 1000, print_cost = False):
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
    K -- number of subgroups for MOM
    num_epochs -- number of epochs of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    costs -- list of costs F calculated at each iteration of the optimization loop
    X -- samples of the learned distribution mu, array of size ( n_y, n_x ) 
    """

    tic = time.time()
    #np.random.seed(1)
    costs = []   # keep track of cost
    errs = []
    costs_mom = []   # keep track of cost
    errs_mom = []

    #seed = 1 
    
    ( d_z , n_z ) = Z.shape
    
    d_y = Y.shape[0]
    
    
    #seed = 1 # for constructing minibatches
    
    # Parameters initialization.
    (C, W, b) = initialize_parameters( d_z, d_y, n_n )
    (C_mom, W_mom, b_mom) = initialize_parameters( d_z, d_y, n_n )
    
    diff_vars = []
    
    N_batch = np.floor(n_z/minibatch_size) 
    
    cost_mini = np.zeros(int(N_batch))
    err_mini = np.zeros(int(N_batch))
    
    cost_mini_mom = np.zeros(int(N_batch))
    err_mini_mom = np.zeros(int(N_batch))
    
    # Loop (optimization)
    for i in range(0, num_epochs): 
        
        for j in range(0, int(N_batch)):
            
            mini_Z = Z[:, (j*minibatch_size) : ((j+1)*minibatch_size) ]
            mini_Y = Y[:, (j*minibatch_size) : ((j+1)*minibatch_size) ]
            
            # Forward propagation.
            (cum_cost, C, W, b)  = forward_propagation( C, W, b, mini_Z, mini_Y, dt, lam, sigma)
            cost_mini[j] = cum_cost
            (cum_cost_mom, C_mom, W_mom, b_mom)  = forward_propagation_MOM( C_mom, W_mom, b_mom, mini_Z, mini_Y, dt, lam, sigma, K )
            cost_mini_mom[j] = cum_cost_mom
            
            diff_vars.append(np.var(b , axis=0))
            
            err_mini[j] = validation(ZV, YV, C, W, b)
            err_mini_mom[j] = validation(ZV, YV, C_mom, W_mom, b_mom)
              
        
        costs.append(np.mean(cost_mini))
        errs.append(np.mean(err_mini))
        
        costs_mom.append(np.mean(cost_mini_mom))
        errs_mom.append(np.mean(err_mini_mom))
        
        if print_cost:
            print ("Cost after epoch %i: %f" %(i, costs[i]))
            print ("validation error %i: %f" %(i, errs[i]))
            
   


    print(time.time() - tic)


    plt.plot(np.squeeze(costs), label='Training')
    plt.plot(np.squeeze(errs), 'r--', label='Validation')
    plt.plot(np.squeeze(costs_mom), 'green', label='Training_MOM')
    plt.plot(np.squeeze(errs_mom), 'y--', label='Validation_MOM')
    plt.legend()
    
   # plt.plot(np.squeeze(diff_vars),'brown')
    return ( C, W, b, costs, C_mom, W_mom, b_mom, costs_mom)


# In[20]:


#regression for a simple function: training errors

#Z = np.random.exponential( 1, ( 1, 1000 ) )
Z =  np.linspace( 0 , 20.0, 200 )
Z =  Z.reshape(1, 200)               
Y =  np.sin(Z)

ZV = np.random.rand(1, 10) * 5
YV = np.sin(ZV)

n_n = 4000

(C, W, b, costs, C_mom, W_mom, b_mom, costs_mom) = nn_model( Z, Y, ZV, YV,  n_n, minibatch_size = 50, dt =0.03, lam = 0.1, sigma = 0.4, K=10,  num_epochs =50)




# In[ ]:

LZ = np.matmul( W, Z ) + b       #dimension: n_n, d_y, n_z
AZ = relu( LZ )  
RZ = np.mean(AZ * trun(C) , axis=0)

LZ_mom = np.matmul( W_mom, Z ) + b_mom       #dimension: n_n, d_y, n_z
AZ_mom = relu( LZ_mom )  
RZ_mom = np.mean(AZ_mom * trun(C_mom) , axis=0)


fig = plt.figure( figsize = (8,3) )
ax0 = fig.add_subplot( 1, 2, 1 )
ax0.plot(  np.squeeze(RZ), label='Learning'  ) 
ax0.plot(  np.squeeze(RZ_mom), 'red' , label='Learning_MOM' ) 
ax0.plot(np.squeeze(Y))
ax0.legend()
    

#Move = np.sum(W*W) / n_n

#print(Move)


# Test on MNIST
#from mlxtend.data import loadlocal_mnist
#
#(z, y) = loadlocal_mnist(
#        images_path='MNIST/train-images-idx3-ubyte', 
#        labels_path='MNIST/train-labels-idx1-ubyte')
#
#Z = z * 1.0 / 255.0
#

#
#enc = OneHotEncoder( sparse = False ) 
#Y = enc.fit_transform( y.reshape( 60000, 1 ) )
#
#(C, W, b, costs) = nn_model( Z.T, Y.T, n_n = 50, minibatch_size = 100, dt =0.1, lam = 0.1, sigma = 0.4, num_epochs = 5)
#



