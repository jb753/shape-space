"""Some test stuff for shape-space."""
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

def bernstein( x, n, i ):
    return binom(n,i) * x**i *(1.-x)**(n-i)

def z_to_shape_space( x, z, zte ):
    """Transform real coordinates to shape space."""
    return (z - x*zte) / np.sqrt(x) / (1. - x)

def z_from_shape_space( x, s, zte ):
    """Transform shape space to real coordinates."""
    return np.sqrt(x) * (1. - x) * s + x*zte

def evaluate_coeffs( x, A ):
    n = len(A)-1
    return np.sum(np.stack([A[i] * bernstein( x, n , i ) for i in range(0,n+1)]),0)

def fit_coeffs( x, s, order ):
    n = order - 1
    X = np.stack( [ bernstein( x, n, i ) for i in range(0,n+1) ] ).T
    return lstsq(X, s, rcond=None)[:2]

def resample_coeffs( A, order ):
    x = np.linspace(0.,1.)
    s = evaluate_coeffs( x, A )
    return fit_coeffs( x, s, order )


if __name__=="__main__":
    x = np.linspace(0,1.)
    s = - (x**3 - x**2. + .2*x)
    A1, _ = fit_coeffs( x, s, 4 )
    A2, _ = resample_coeffs( A1, 3 )
    A3, _ = resample_coeffs( A1, 10 )
    y = evaluate_coeffs( x, A3 )
    f, a = plt.subplots()
    a.plot(x,s,'-')
    a.plot(x,y,'o')
    plt.show()
