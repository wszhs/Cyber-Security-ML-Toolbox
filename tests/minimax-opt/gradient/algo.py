import autograd
import os
from autograd import grad
from autograd import jacobian
import numpy as np
from scipy.linalg import pinv

# GDA
def gda(z_0, alpha=0.05, num_iter=100,target=None):
    z = [z_0]
    grad_fn = grad(target)
    print(z[-1])
    for i in range(num_iter):
        g = grad_fn(z[-1])
        z1 = z[-1] + g*np.array([-1,1])*alpha
        z.append(z1)
    z = np.array(z)
    return z

# Extra Gradient
def eg(z_0, alpha=0.05, num_iter=100,target=None):
    z = [z_0]
    grad_fn = grad(target)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        z1 = z[-1] + g*np.array([-1,1])*alpha
        g = grad_fn(z1)
        z2 = z[-1] + g*np.array([-1,1])*alpha
        z.append(z2)
    z = np.array(z)
    return z

# Optimistic Gradient
def ogda(z_0, alpha=0.05, num_iter=100,target=None):
    z = [z_0,z_0]
    grads = []
    grad_fn = grad(target)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        gg = grad_fn(z[-2])
        z1 = z[-1] + 2*g*np.array([-1,1])*alpha - gg*np.array([-1,1])*alpha
        z.append(z1)
    z = np.array(z)
    return z

# Consensus Optimization
def co(z_0, alpha=0.01, gamma=0.1, num_iter=100,target=None):
    z = [z_0]
    grads = []
    grad_fn = grad(target)
    hessian = jacobian(grad_fn)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        H = hessian(z[-1])
        #print(np.matmul(H,g), z[-1])
        v = g*np.array([1,-1]) + gamma*np.matmul(H,g)
        z1 = z[-1] - alpha*v
        z.append(z1)
    z = np.array(z)
    return z

# Symplectic gradient adjustment
def sga(z_0, alpha=0.05, lamb=0.1, num_iter = 100,target=None):
    z = [z_0]
    grad_fn = grad(target)
    hessian = jacobian(grad_fn)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        w = g * np.array([1,-1])
        H = hessian(z[-1])
        HH = np.array([[1, -lamb*H[0,1]],[lamb*H[0,1],1]])
        v = HH @ w
        z1 = z[-1] - alpha*v
        z.append(z1)
    z = np.array(z)
    return z

# Follow the ridge
def follow(z_0, alpha=0.05, num_iter = 100,target=None):
    z = [z_0]
    grad_fn = grad(target)
    hessian = jacobian(grad_fn)
    for i in range(num_iter):
        g = grad_fn(z[-1])
        H = hessian(z[-1])
        v = np.array([g[0], -g[1]-H[0,1]*np.squeeze(pinv(H[1:,1:]))*g[0]])
        z1 = z[-1] - alpha*v
        z.append(z1)
    z = np.array(z)
    return z
