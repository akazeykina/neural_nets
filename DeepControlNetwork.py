
# coding: utf-8

# In[1]:


import numpy as np
#import h5py
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d.axes3d import get_test_data # for 3d plotting
#import utils as ut # file containing a function to construct minibatches 

#get_ipython().run_line_magic('matplotlib', 'inline')

#np.random.seed(1)


# In[15]:


def relu( x ):
    x = np.maximum(-5,x)
    return np.minimum( 5, x )
#    return np.maximum(0,x)

def trun(x) :   
    x = np.maximum(-100,x)
    return np.minimum( 100, x )

# In[16]:


def drelu( x ):
    return np.where( (x > -5) & (x < 5) , 1, 0 )
#    return np.where(x>0, 1, 0)



def dtrun( x ):
    return np.where( (x > -100) & (x < 100) , 1, 0 )

# In[4]:


def initialize_parameters( n_y, n_n, num_layers):
    """
    Arguments:
    n_z -- the number of features in the input data
    n_y -- the number of features of the output data
    n_n -- number of neurons in each layer
    L -- the number of layers
    
    Returns:
    parameters -- python dictionary containing parameters "C1", "W1", "B1" "b1", ..., "CL", "WL", "BL", "bL":
                    C -- weight matrix of shape (n_n, n_y, n_y)
                    W -- weight matrix of shape (n_n, n_y, n_y)
                    b -- bias vector of shape (n_n, n_y, 1)
    """
    norm = 12
    
    C = np.random.randn(n_n, n_y, 1, num_layers-1) * norm
    W = np.random.randn(n_n, n_y, n_y, num_layers-1)* norm
    b = np.random.randn(n_n, n_y, 1, num_layers-1)* norm
        
        
    return (C, W, b)


# In[5]:


def forward_propagation(z, C, W, b, num_layers, dt):
    """
    Implement forward propagation for L layers (see formula for dX)
    
    Arguments:
    z -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters()
    L -- number of layers
    dt -- parameter of gradient update rule
    
    Returns:
    caches -- list of caches containing:
                the output of the linear unit and of the nonlinear activation of every layer
                (there are L-1 of them, indexed from 1 to L-1)
    """
    
    n_y = b.shape[ 1 ]

    m = z.shape[1]
 
    XA = np.zeros((n_y, m, num_layers))
    XA[:,:, 0] = z
    
    
    # Implement dX formula. Add "cache" to the "caches" list.
    for l in range( num_layers - 1):
        
        LX = np.matmul( W[:,:,:, l], XA[:,:,l] ) + b[:,:,:,l] #  dim: nn,ny,m

        dX = np.mean( trun(C[:,:,:,l]) * relu( LX ) , axis = 0 ) * dt
        
        
        XA[:,:,l+1] = dX + XA[:,:,l] 
    
    return XA



# In[7]:


def backward_propagation(XL, Y, XA, C, W, b, num_layers, dt, ds, lam, eps):
    """
    Implement the backward propagation (see formulas for dtheta and dp)
    
    Arguments:
    XL -- output of the forward propagation
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches 
                
    
    Returns:
    parameters -- a dictionary of updated parameters
    
    """
    m = Y.shape[ 1 ]
    n_y = Y.shape[ 0 ]
    n_n = C.shape[ 0 ]
    
    # Initializing the backpropagation
    P =  XA[:,:,num_layers-1 ] - Y 
    
    
    cost = np.sum(P*P)/2/m
    
    # Creating the same delta W for all layers (i.e. all t)
    C_noise = np.random.randn( n_n, n_y, 1 )
    W_noise = np.random.randn( n_n, n_y, n_y )
    b_noise = np.random.randn( n_n, n_y, 1 )
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(num_layers-1)):
        X = XA[:,:, l ]                  #dimension: dy, m
        
        Cl = C[:,:,:, l]
        Wl = W[:,:,:, l]
        bl = b[:,:,:,l]

        LX = np.matmul( Wl, X ) + bl       #dimension: n_n, d_y, m
        AX = relu( LX )  
        dAX = drelu( LX )
        
        # In the update rule for theta should one use P or P_prev?

        BX = trun(Cl) * dAX                   #dimension: n_n, d_y, m

        
        dC = (- lam * Cl - np.mean(AX * P, axis =2, keepdims = True)*dtrun(Cl)) * ds + eps * np.sqrt(ds) * C_noise
        C[:,:,:, l] = Cl + dC
        
                                   
        dW = (- lam * Wl - np.matmul( BX * P , X.T ) / m) * ds + eps * np.sqrt(ds) * W_noise
           
    
        db = (- lam * bl - np.mean( BX * P, axis =2 , keepdims = True) ) * ds + eps * np.sqrt(ds) * b_noise
        b[:,:,:, l] = bl + db
        
        temp = np.transpose(BX.reshape(1, n_n, n_y, m), axes = (1,0,2,3))
        WW   = np.transpose(Wl.reshape(1, n_n, n_y, n_y), axes =(1,3,2,0))
        
        dP =   ( np.sum(np.mean(temp * WW * P, axis = 0), axis = 1) ) * dt
        P = P + dP                      #dimension: d_y, m
        
        W[:,:,:, l] = Wl + dW
        
    
                                   
    return (C, W, b, cost)


# In[8]:
    
def validation(zV, YV, C, W, b, num_layers, dt):
    
    m = zV.shape[1]
    
    F = forward_propagation(zV, C, W, b,  num_layers, dt)

    Dif = F[:,:,-1]  - YV
    
    err = np.sum(Dif * Dif)/2/m
        

    return err

######



def deep_nn_model(z, Y, zV, YV, num_layers, n_n, dt = 0.01, ds = 0.01, lam = 0.1, eps = 0.1, 
                  num_epochs = 1000, print_cost= False ):
    """
    Implements a L-layer neural network.
    
    Arguments:
    z -- data, numpy array of shape (number of features, number of examples)
    Y -- true "label" vector, of shape (n_y, number of examples)
    num_layers -- number of layers
    n_n -- number of neurons in each layer
    minibatch_size -- size of minibatch
    dt -- parameter of the gradient update rule
    lam -- parameter of the gradient update rule
    eps -- parameter of the gradient update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    #np.random.seed(1)
    costs = []   # keep track of cost
    
    errs = []
    
    Lvars =[]
    
    n_y = Y.shape[ 0 ]
    
    #seed = 1 # for constructing minibatches
    
    # Parameters initialization.
    (C, W, b) = initialize_parameters( n_y, n_n, num_layers )
    
    it_num = 0
    
    err = 0
    cost10 = 0
    
    # Loop (optimization)
    for i in range(num_epochs):
                
        # Forward propagation.
        XA = forward_propagation( z, C, W, b, num_layers, dt )
            
        XL = XA[:,:,num_layers-1]
                
        # Backward propagation.
        (C, W, b, cost) = backward_propagation( XL, Y, 
                                                  XA, C, W, b, num_layers, dt, ds, lam, eps )
            
        err += validation(zV, YV, C, W, b, num_layers, dt)
        cost10 += cost
        
        Lvars.append(np.squeeze(np.var(C[:,:,:,1], axis = 0)))
        
        if np.mod(i+1,10) == 0:
            costs.append(cost10/10) 
        
            errs.append(err/10)
            
            err = 0
            cost10 = 0
            
            
        it_num = it_num + 1
            
        # Print the cost every 100 training example
        if print_cost and it_num % 100 == 0:
            print ("Cost after iteration %i: %f" %(it_num, cost))
            print ("validation error %i: %f" %(it_num, err))
                



        
    
            
    # plot the cost
    plt.figure( figsize = (5,3) )
    plt.plot(  np.log(costs) , 'blue', label = 'training error' ) 
    plt.plot(  np.log(errs) , 'red', label = 'validation error')
    plt.ylabel('errors')
    plt.xlabel('iteration*10')
    plt.title("Log errors: 100 neurons per layer")
    plt.legend()
    
    
    plt.figure( figsize = (5,3) )
    plt.plot(  Lvars  ) 
    plt.xlabel('iteration')
    plt.ylabel('variance')
    plt.title('Variance of neurons: 10 neurons')
    
    return (C, W, b)


# In[14]:


#regression for a simple function: training errors

#Z = np.random.exponential( 1, ( 1, 1000 ) )
z =  np.linspace( 0 , 10.0, 20 )
z =  z.reshape(1, 20)               
Y =  np.sin(z)

zV = np.random.rand(1, 100) * 10
YV = np.sin(zV)

num_layers =4
dt = 0.05
n_n= 32

(C, W, b) = deep_nn_model( z, Y,zV, YV, num_layers , n_n , dt , ds = 0.04, lam = 0.0, eps = 0.1, 
                  num_epochs = 2000 )


# In[10]:


F = forward_propagation(z, C, W, b,  num_layers, dt= 0.05)

FL = F[:,:,-1]


fig = plt.figure( figsize = (8,3) )
ax0 = fig.add_subplot( 1, 2, 2 )
ax0.plot(  np.squeeze(FL)  ) 
ax0.plot(np.squeeze(Y))

 
Move = np.sum(W*W) * dt / n_n

print(Move)


# In[ ]:

# Test on MNIST
#from mlxtend.data import loadlocal_mnist
#
#z, y = loadlocal_mnist(
#        images_path='MNIST/train-images-idx3-ubyte', 
#        labels_path='MNIST/train-labels-idx1-ubyte')
#
#from sklearn.preprocessing import OneHotEncoder
#
#enc = OneHotEncoder( sparse = False )
#y_train = y.reshape( 60000, 1 )
#y_train = enc.fit_transform( y_train )
#y_train = y_train.T

