import numpy as np
import jax.numpy as jnp
from numpy.polynomial.legendre import leggauss

def leggaussVarLimits(n, a, b, dtype=jnp.double):
    """
    Computers n weights and points for Gauss-Legendre numerical integration of a
    function in the interval (a,b).
    """
    x, w = leggauss(deg=n)
    # reversing x and w to make them compatible with Lea's Matlab legquad function
    # x = x[::-1].copy()
    # w = w[::-1].copy()
    x = jnp.asarray(x)
    w = jnp.asarray(w)
    xVarLimits = (x*(b-a)+(b+a))/2
    wVarLimits = (b-a)/2*w
    return xVarLimits, wVarLimits

def gram_schmidt(A):
    '''input: A set of linearly independent vectors stored
              as the columns of matrix A
       outpt: An orthongonal basis for the column space of A.'''
    # get the number of vectors.
    A = np.copy(A).astype(np.float64) # create a local instance of the array
    n = A.shape[1]
    for j in range(n):
        # For the vector in column j, find the perpendicular
        # of the projection onto the previous orthogonal vectors.
        for k in range(j):
            A[:, j] -= np.dot(A[:, k], A[:, j]) * A[:, k]
        # If original vectors aren't lin indep then we can check for this:
        # 
        if np.isclose(np.linalg.norm(A[:, j]), 0, rtol=1e-15, atol=1e-14, equal_nan=False):
            A[:, j] = np.zeros(A.shape[0])
        else:    
            A[:, j] = A[:, j] / np.linalg.norm(A[:, j])
    return A
