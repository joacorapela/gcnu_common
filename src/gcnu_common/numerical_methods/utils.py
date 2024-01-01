import pdb
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
