
# coding: utf-8

# In[14]:


import numpy as np
import h5py
import matplotlib.pyplot as plt
import time


get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


def relu( x ):
    x = np.maximum(-5,x)
    return np.minimum( 5, x )
    # return np.maximum(0,x)

def trun(x) :
    x = np.maximum(-20,x)
    return np.minimum( 20, x )

# In[16]:


def drelu( x ):
    return np.where( (x > -5) & (x < 5) , 1, 0 )
#    return np.where(x>0, 1, 0)

def dtrun( x ):
    return np.where( (x > -20) & (x < 20) , 1, 0 )

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
    
    norm = 15
    C = (np.random.rand( n_n, d_y , 1) -1/2) * norm
    W = (np.random.rand( n_n, d_y , d_z) -1/2) * norm
    b = (np.random.rand( n_n, d_y , 1) -1/2)  * norm
    
    p = np.ones( n_n)
    p = p/n_n

        
    return ( C, W, b, p )


# In[18]:



def forward_propagation( C, W, b, p, Z, Y, dt, lam, sigma ):
    """
    Implement forward propagation (see formulas for dX, dTheta)
    
    Arguments:
    parameters -- current value of Theta = ( C, W, b )
    Z -- data, array of size ( d_z, n_z )
    Y -- label, array of size ( d_y, n_z)
    dt, lam, sigma -- parameters of gradient update rule
    p is the weight
    
    Returns:

    parameters -- updated value of parameter Theta
    """
    
    (d_z, n_z) = Z.shape
    d_y = Y.shape[ 0 ]
    n_n = C.shape[ 0 ]
        
    
    LZ = np.matmul( W, Z ) + b       #dimension: n_n, d_y, n_z
    AZ = relu( LZ )  
    
    # RZ = np.mean(AZ * trun(C) , axis=0)
    RZ = np.average(AZ * trun(C) , axis=0, weights = p)
    
    dAZ = drelu( LZ )
    BZ = trun(C) * dAZ
    
    
    diff =  RZ  - Y       # dimension:  d_y, n_z

    
#    diff = diff + np.where(RZ>1, 0.5, 0) + np.where(RZ<-1,-0.5,0)
       
#    diff =   diff.reshape(1, d_y, n_z)
    
    # Implement dTheta formula
    dc_dF_dnu = np.mean( AZ * diff, axis = 2, keepdims = True ) * dtrun(C)        # dimension:  n_n, d_y, 1
    
    dw_dF_dnu = np.matmul(BZ * diff, Z.T)/n_z               # dimension:  n_n, d_y, d_z
    db_dF_dnu = np.mean( BZ * diff , axis = 2, keepdims = True )        # dimension:  n_n, d_y, 1
    
    dC = - ( dc_dF_dnu + lam * C ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, 1 ) 
    Cu = C + dC
    
    dW = - ( dw_dF_dnu + lam * W ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, d_z )
    Wu = W + dW
    
    db = - ( db_dF_dnu + lam * b ) * dt + sigma * np.sqrt(dt) * np.random.randn( n_n, d_y, 1 )
    bu = b + db
    

    
    dF1 = np.mean(np.sum( trun(Cu) * relu( np.matmul( Wu, Z ) + bu  ) * diff, axis = 1 ), axis = 1 )
    dF0 = np.mean(np.sum( trun(C) * AZ * diff, axis = 1 ), axis = 1 )
    dF = -2/sigma/sigma * ( dF1.reshape(1,n_n) - np.transpose(dF0.reshape(1,n_n), axes=(1,0)) )
    # dF = np.minimum(dF, 200)                                                        
    
    dC = np.linalg.norm(C.reshape(1, n_n, d_y , 1) - np.transpose(Cu.reshape(1, n_n, d_y, 1), axes=(1,0,2,3)), axis=(2,3))
    dW = np.linalg.norm(W.reshape(1, n_n, d_y , d_z) - np.transpose(Wu.reshape(1, n_n, d_y, d_z), axes=(1,0,2,3)), axis=(2,3))
    db = np.linalg.norm(b.reshape(1, n_n, d_y , 1) - np.transpose(bu.reshape(1, n_n, d_y, 1), axes=(1,0,2,3)), axis=(2,3))
    
    sigma0 = sigma
    A =  np.exp(dF -1/sigma0 /sigma0 /dt * (dC**2 + dW**2 + db**2))
    # print ("df", delta_F)
    
    # A = np.zeros([n_n, n_n])
    
    # for i in range(0, n_n):
    #     for j in range(0, n_n):
    #         ddC = np.linalg.norm(C[j,:,:] - Cu[i,:,:])
    #         ddW = np.linalg.norm(W[j,:,:] - Wu[i,:,:])
    #         ddb = np.linalg.norm(b[j,:,:] - bu[i,:,:])
    #         A[i,j] = np.exp(-2/sigma/sigma * (dF1[i] - dF0[j]  + 1/dt/2 * ( ddC**2 + ddW**2 + ddb**2 ) )) 
    #         # A[i,j] = np.exp(-2/sigma/sigma * delta_F[i] )
   
    p = p/np.sum(A, axis=0)
    A = A * p
    
    p = np.sum(A, axis= 1)
    
    # print(p)
    
    
    cost = np.sum(diff * diff)/2/n_z
    
    cost = np.squeeze(cost)      # To make sure cost's shape is what we expect (e.g. this turns [[17]] into 17).
    
    return (cost, Cu, Wu, bu, p)


# In[19]:


def validation(ZV, YV, C, W, b):
    
    m = ZV.shape[1]
    
    LZ = np.matmul( W, ZV ) + b       #dimension: n_n, d_y, n_z
    AZ = relu( LZ )  
    Dif = np.mean(AZ * trun(C) , axis=0) - YV
    
    err = np.sum(Dif * Dif)/2/m
        

    return err



def nn_model(Z, Y, ZV, YV, n_n, minibatch_size = 32, dt = 0.1, lam = 0.01, sigma = 0.01, 
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

    #seed = 1 
    
    ( d_z , n_z ) = Z.shape
    
    d_y = Y.shape[0]
    
    
    #seed = 1 # for constructing minibatches
    
    # Parameters initialization.
    (C, W, b ,p) = initialize_parameters( d_z, d_y, n_n )
    
    diff_vars = []
    
    N_batch = np.floor(n_z/minibatch_size) 
    
    cost_mini = np.zeros(int(N_batch))
    err_mini = np.zeros(int(N_batch))
    
    # Loop (optimization)
    for i in range(0, num_epochs): 
        
        for j in range(0, int(N_batch)):
            
            mini_Z = Z[:, (j*minibatch_size) : ((j+1)*minibatch_size) ]
            mini_Y = Y[:, (j*minibatch_size) : ((j+1)*minibatch_size) ]
            
            # Forward propagation.
            (cum_cost, C, W, b, p)  = forward_propagation( C, W, b, p,  mini_Z, mini_Y, dt, lam, sigma )
            cost_mini[j] = cum_cost
            
            diff_vars.append(np.var(b , axis=0))
            
            err_mini[j] = validation(ZV, YV, C, W, b)
              
        
        costs.append(np.mean(cost_mini))
        errs.append(np.mean(err_mini))
        
        if print_cost:
            print ("Cost after epoch %i: %f" %(i, costs[i]))
            print ("validation error %i: %f" %(i, errs[i]))
            
   


    print(time.time() - tic)

    print(p)
    plt.plot(np.squeeze(costs))
    plt.plot(np.squeeze(errs), 'red')
    # plt.plot(np.squeeze(diff_vars),'brown')
    return ( C, W, b, costs)


# In[20]:


#regression for a simple function: training errors

#Z = np.random.exponential( 1, ( 1, 1000 ) )
Z =  np.linspace( 0 , 1, 50 )
Z =  Z.reshape(1, 50)               
Y =  np.where((Z>0.3) & (Z<0.4),1,0)

ZV = np.random.rand(1, 100)
YV = np.where((ZV>0.3) & (ZV<0.4),1,0)

n_n = 200

(C, W, b, costs) = nn_model( Z, Y, ZV, YV,  n_n, minibatch_size = 50, dt =0.05, lam = 0.001, 
                            sigma = 0.2, num_epochs =2000)




# In[ ]:

LZ = np.matmul( W, Z ) + b       #dimension: n_n, d_y, n_z
AZ = relu( LZ )  
RZ = np.mean(AZ * trun(C) , axis=0)

fig = plt.figure( figsize = (8,3) )
ax0 = fig.add_subplot( 1, 2, 1 )
ax0.plot(  np.squeeze(RZ)  ) 
ax0.plot(np.squeeze(Y))
    

Move = np.sum(W*W) / n_n

print(Move)


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



