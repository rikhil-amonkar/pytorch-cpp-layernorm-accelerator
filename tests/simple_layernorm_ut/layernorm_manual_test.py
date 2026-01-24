import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

# create sample data input tensors (2D data)
n_samples, embedding_dim = 1, 512
x_2d = np.random.randn(n_samples, embedding_dim)
gamma = np.ones(embedding_dim)
beta = np.zeros(embedding_dim)

# create forwards pass function (tensor, gamma, beta, epsilon)
def forward(x, gamma, beta, epsilon=1e-5):
    
    # get the dimensions of the current tensor
    n, dims = x.shape
    
    # calculate mean of input tensor (across dims)
    mu = (1 / dims) * np.sum(x, axis=-1, keepdims=True)  # last dimension (layer norm)
    
    # calculate the mean centered input
    xmu = x - mu
    
    # calculate the squared deviation (square of centered means)
    sq = xmu ** 2
    
    # calculate the variance (mean of squared deviations across dims)
    var = (1 / dims) * np.sum(sq, axis=-1, keepdims=True)
    
    # add numerical stability with epsilon constant
    sqrtvar = np.sqrt(var + epsilon)  # prevent division by zero
    
    # invert adjusted squared variance
    ivar = 1 / sqrtvar
    
    # execute normalization
    xhat = xmu * ivar
    
    # apply learnable parameters
    out = (gamma * xhat) + beta
    
    # store intermediate values in cache
    cache = (xhat, gamma, xmu, ivar, sqrtvar, var, epsilon)
    
    return out, cache

# create backwards pass function (forward output, forward cache)
def backward(dout, cache): 
    
    # unpack intermediate values from cache
    xhat, gamma, xmu, ivar, sqrtvar, var, epsilon = cache
    
    print(xhat.shape, gamma.shape, xmu.shape, ivar.shape, sqrtvar.shape, var.shape)
    
    # get dimensions of input/output (identical)
    n, dims = dout.shape
    
    # calculate gradient beta using relation to output beta
    dbeta = np.sum(dout, axis=0)  # sum across samples
    dgammax = dout
    
    # calculate gradient gamma using relation to output gamma
    dgamma = np.sum(dgammax * xhat, axis=0)
    dxhat = dgammax * gamma  # gradient flows through scaling
    
    # calculate gradient variance using relation to output inverse variance
    divar = np.sum(dxhat * xmu, axis=-1, keepdims=True)
    dxmu1 = dxhat * ivar  # first component of gradient
    
    # calculate gradient squared variance using chain rule
    dsqrtvar = (-1 / (sqrtvar ** 2)) * divar
    
    # calculate gradient variance using chain rule
    dvar = 0.5 * (1 / np.sqrt(var + epsilon)) * dsqrtvar
    
    # calculate squared deviations using output variance
    dsq = (1 / dims) * np.ones((n, dims)) * dvar
    
    # calculate second component of gradient
    dxmu2 = 2 * xmu * dsq
    
    # combine gradient components and get gradient mean
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dx1, axis=-1, keepdims=True)
    
    # calculate gradient component from mean compuation
    dx2 = (1 / dims) * np.ones((n, dims)) * dmu
    
    # calculate final gradient by combining components
    dx = dx1 + dx2
    
    return dx, dgamma, dbeta



