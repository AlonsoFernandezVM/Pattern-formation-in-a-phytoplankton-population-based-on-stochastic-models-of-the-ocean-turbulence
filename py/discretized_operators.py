import numpy as np
from scipy.ndimage import map_coordinates


def discrete_gradient(X, h):
    '''Function that computes the gradient approximation of a 2D array with the
    fourth order centered formula'''

    grad_x = (np.roll(X, 2, axis = 1) - 8 * np.roll(X, 1, axis = 1) + \
                    8 * np.roll(X, -1 ,axis = 1) - np.roll(X, -2 ,axis = 1)) / \
                    (12 * h)

    grad_y = (np.roll(X, 2, axis = 0) - 8 * np.roll(X, 1, axis = 0) + \
                    8 * np.roll(X, -1 ,axis = 0) - np.roll(X, -2 ,axis = 0)) / \
                    (12 * h)

    return grad_x, grad_y

def discretized_divergence(X, h):
    '''Function that approximates the divergence of a 2D array'''

    X_x, X_y = discrete_gradient(X, h)

    return X_x + X_y

def discrete_laplacian(X, h):
    '''Function that approximates the Laplacian of a 2D array'''

    x = (-np.roll(X,-2,axis = 0) + 16 * np.roll(X,-1,axis = 0) - 30 * X + 16 * np.roll(X,1,axis = 0) - np.roll(X,2,axis = 0) - 
            np.roll(X,-2,axis = 1) + 16 * np.roll(X,-1,axis = 1) - 30 * X + 16 * np.roll(X,1,axis = 1) - np.roll(X,2,axis = 1)) / (12*h**2)

    return x

def semi_lagrangian_tendency(X, h, dt, u, v, order=3):
    '''Function that interpolates the semi lagrangian tendency of a flow in the interior 
    of a 2D array given a (u, v) velocity of a flow'''
    
    tend = np.zeros_like(X)

    phi = X
    Ny, Nx = phi.shape

    # index coords for the interior subarray (0..N-1)
    rows, cols = np.indices((Ny, Nx), dtype=np.float64)

    # backtrace in *index* units: shift = velocity * dt / h
    cols_b = cols - (u * dt / h)
    rows_b = rows - (v * dt / h)

    phi_back = map_coordinates(phi, [rows_b, cols_b],
                               order=order, mode='wrap', prefilter=(order > 1))

    tend[1:-1,1:-1] = (phi_back[1:-1,1:-1] - phi[1:-1,1:-1]) / dt
    return tend