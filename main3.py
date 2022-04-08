import numpy as np
import matplotlib.pyplot as plt

# RNG
rng = np.random.default_rng()

# Fix global variables
n_z     = 2   # number of features
n_n     = 1000  # number of initial neurons
n_per_dim = 11
n       = n_per_dim**n_z
sigma   = 0.3
alpha   = 1.0

def reLU(x):
    return np.maximum(0,x)

def dreLU(x):
    return np.where(x > 0, 1, 0)

def loss(x, y):
    return (x - y)**2 / 2

def loss_vec(
    xs: np.ndarray((n)),
    ys: np.ndarray((n))
    ):
    return (xs - ys)**2 / 2

def predict(
    W: np.ndarray((n_n, n_z)), 
    b: np.ndarray((n_n)), 
    z: np.ndarray((n_z)),
    weights: np.ndarray((n_n)),
    activ_func: "function" = reLU
    ):
    '''
    Implement
    y = sum_{i=1}^{n_n} 1/n_n * phi(sum_{j=1}^{n_z} z_j W_{ij} + b_i)

    Here we assume activ_func is "vectorized", i.e. allowing passing
    array_like object as argument, and computes component-wise. 
    '''
    activ_func_arg = np.matmul(W, z) + b # of shape (n_n)
    return np.dot(activ_func(activ_func_arg), weights)

def predict_vec(
    W: np.ndarray((n_n, n_z)), 
    b: np.ndarray((n_n)), 
    weights: np.ndarray((n_n)),
    zs: np.ndarray((n_z, n)),
    activ_func: "function" = reLU
    ):
    '''
    Implement vectorized version of predict(). 
    '''
    activ_func_arg = np.tensordot(W, zs, axes=(1,0)) + np.expand_dims(b, axis=1) # of shape (n_n,n)
    return np.tensordot(activ_func(activ_func_arg), weights, axes=(0,0)).squeeze()

def init_params(mean=0, scale=1):
    '''
    Initialize W and b. 
    '''
    W = scale * rng.standard_normal(size=(n_n,n_z)) + mean
    b = scale * rng.standard_normal(size=(n_n)) + mean
    weights = np.full((n_n), 1.0/n_n)
    return W, b, weights

def init_FP_params(mean=0, scale=1):
    '''
    Initialize W and b for the Fokker-Planck dynamics.
    '''
    w_0 = scale * rng.standard_normal(size=(n_z)) + mean # of shape (n_z)
    b_0 = scale * rng.standard_normal() + mean           # of shape ()
    return w_0, b_0

def get_mstar(
    W: np.ndarray((n_n, n_z)), 
    b: np.ndarray((n_n)),
    weights: np.ndarray((n_n)),
    zs: np.ndarray((n_z, n)),
    ys: np.ndarray((n)),
    activ_func: "function" = reLU,
    d_activ_func: "function" = dreLU,
    horizon_s: float = 10,
    delta_s: float = 0.1,
    ):
    '''
    Calculate for m given the corresponding m^* using the Fokker-Planck
    dynamics. 
    '''
    # Initialize
    num_epochs = int(np.floor(horizon_s / delta_s))
    Wstar = np.ndarray((num_epochs, n_z))
    bstar = np.ndarray((num_epochs))
    w_0, b_0 = init_FP_params() # of shape (n_z) and () respectively
    Wstar[0] = w_0
    bstar[0] = b_0

    predicts = predict_vec(W, b, weights, zs, activ_func)

    for epoch_s in range(1, num_epochs):
        w = Wstar[epoch_s-1] # of shape (n_z)
        b = bstar[epoch_s-1] # of shape ()

        activ_func_arg = np.tensordot(w, zs, axes=(0,0)) + np.expand_dims(b, axis=0) # of shape (n)
        d_activ_func_output = d_activ_func(activ_func_arg) # of shape (n)

        b_drift = - np.dot(
            predicts - ys, # of shape (n)
            d_activ_func_output # of shape (n)
            )
        # Probably suboptimal
        w_drift = np.ndarray((n_z))
        for j in range(n_z):
            w_drift[j] = - ((predicts - ys) * d_activ_func_output * zs[j]).sum()

        # Increment 
        w = w + w_drift * delta_s + sigma * np.sqrt(delta_s) * rng.standard_normal(size=(n_z))
        b = b + b_drift * delta_s + sigma * np.sqrt(delta_s) * rng.standard_normal()

        Wstar[epoch_s] = w
        bstar[epoch_s] = b

    return Wstar, bstar, np.full((num_epochs), 1.0/num_epochs)

def mix(Ws1, bs1, weights1, Ws2, bs2, weights2, l):
    '''
    Get the linear combination of two empirical measures.
    '''
    #n = len(weights1) + len(weights2)
    Ws = np.append(Ws1, Ws2, axis=0)
    bs = np.append(bs1, bs2, axis=0)
    weights = np.append(weights1 * (1 - l), weights2 * l)
    return Ws, bs, weights

def error(Ws, bs, weights, zs, ys, activ_func=reLU):
    predicts = predict_vec(Ws, bs, weights, zs, activ_func)
    return loss(predicts, ys).mean()

if __name__ == "__main__":
    target_func = lambda x_1, x_2: np.sin(2 * x_1) + np.cos(2* x_2)
    #zs, ys = gen_samples(target_func, 0, 1)

    # generate samples
    def gen_samples(func, interval_start, interval_end, size=n_per_dim):
        '''
        Generate samples from vectorized function func()
        '''
        zs = np.linspace(interval_start, interval_end, size)
        ys = func(zs)
        return zs, ys

    def gen_2d_samples(func, interval_start, interval_end, size=n_per_dim):
        coordinate_list = np.linspace(interval_start, interval_end, size)
        z1s, z2s = np.meshgrid(coordinate_list, coordinate_list)
        z1s = z1s.flatten()
        z2s = z2s.flatten()
        zs = np.array([z1s, z2s])
        ys = func(z1s, z2s)
        return zs, ys

    zs, ys = gen_2d_samples(target_func, 0, 1)

    zs_valid, ys_valid = gen_2d_samples(target_func, 0, 1, size=51)

    W, b, weights = init_params()

    horizon_t = 10.0
    delta_t = 0.2
    num_epochs = int(np.floor(horizon_t / delta_t))

    # Train
    for epoch_t in range(num_epochs):
        if (epoch_t % 10 == 0):
            print("EPOCH", epoch_t)
            print(error(W, b, weights, zs, ys))
            print(error(W, b, weights, zs_valid, ys_valid))
            print(len(weights))

        Wstar, bstar, weights_star = get_mstar(W, b, weights, zs, ys)
        W, b, weights = mix(W, b, weights, Wstar, bstar, weights_star, alpha*delta_t)

    predicts = predict_vec(W, b, weights, zs)

    '''

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(zs[0].reshape(21,21), zs[1].reshape(21,21), ys.reshape(21,21), 50, cmap='binary')
    ax.contour3D(zs[0].reshape(21,21), zs[1].reshape(21,21), predicts_1.reshape(21,21), 50, cmap='binary')
    ax.contour3D(zs[0].reshape(21,21), zs[1].reshape(21,21), predicts.reshape(21,21), 50, cmap='binary')
    ax.set_xlabel('z1')
    ax.set_ylabel('z2')
    ax.set_zlabel('y');
    
    plt.show()
    '''
