"""Some test stuff for shape-space."""
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

dx = 0.04

def read_seglid(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()[1:]
    # Skip header
    xy = np.array([[float(i) for i in line.strip().split()] for line in lines]).T
    # Find the turning point of x coordinate
    i1 = np.where(np.diff(xy[0,:])>0.)[0][0]
    # Determine axial chord and normalise
    c = np.ptp(xy[0,:])
    # Split into pressure and suction sides, normalise
    xy1 = np.flip(xy[:,:i1],1)/c
    xy2 = xy[:,i1:]/c
    return xy1, xy2

def bernstein( x, n, i ):
    return binom(n,i) * x**i *(1.-x)**(n-i)

def z_to_shape_space( x, z, zte ):
    """Transform real coordinates to shape space."""
    dz_dx = np.gradient(z, x)
    # Preallocate
    s = np.ones(x.shape) * np.nan
    ii = np.abs(x-0.5)<(0.5-dx)
    s[ii] = (z[ii] - x[ii]*zte) / ( np.sqrt(x[ii]) * (1. - x[ii]) )
    return s

def z_from_shape_space( x, s, zte ):
    """Transform shape space to real coordinates."""
    return np.sqrt(x) * (1. - x) * s + x*zte

def evaluate_coeffs( x, A ):
    n = len(A)-1
    return np.sum(np.stack([A[i] * bernstein( x, n , i ) for i in range(0,n+1)]),0)

def fit_coeffs( x, s, order ):
    n = order - 1
    xtrim = x[np.abs(x-0.5)<(0.5-dx)]
    strim = s[np.abs(x-0.5)<(0.5-dx)]
    X = np.stack( [ bernstein( xtrim, n, i ) for i in range(0,n+1) ] ).T
    return lstsq(X, strim, rcond=None)[:2]

def resample_coeffs( A, order ):
    x = np.linspace(0.,1.)
    s = evaluate_coeffs( x, A )
    return fit_coeffs( x, s, order )


if __name__=="__main__":


    xy = read_seglid( 'naca6412.dat' )

    zte = 0.
    order = 4

    s = [z_to_shape_space(*xyi, zte) for xyi in xy]

    # print(fit_coeffs(xy[0][0,:], s[0], order))


    A, res = zip(*[fit_coeffs( xyi[0,:], si, order ) for xyi, si in zip(xy, s) ])


    xfit =  np.linspace(0.,1.,1000)
    sfit = [evaluate_coeffs( xfit, Ai ) for Ai in A]

    yfit = [ z_from_shape_space( xfit, sfiti, zte) for sfiti in sfit]

    # x = np.linspace(0,1.)
    # s = - (x**3 - x**2. + .2*x)
    # A1, _ = fit_coeffs( x, s, 4 )
    # A2, _ = resample_coeffs( A1, 3 )
    # A3, _ = resample_coeffs( A1, 10 )

    f, a = plt.subplots()
    f2, a2 = plt.subplots()
    for xyi, si, sfiti, yfiti in zip(xy, s, sfit, yfit):
        a2.plot(xyi[0,:],si,'-')
        a2.plot(xfit,sfiti,'--')
        a.plot(*xyi,'kx')
        a.plot( xfit, yfiti, 'k-')
    # a.plot(*xy2,'kx')
    a.axis('equal')
    a2.axis('equal')
    plt.show()
