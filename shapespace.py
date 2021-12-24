"""Some test stuff for shape-space."""
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

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
    # Preallocate
    s = np.ones(x.shape) * np.nan
    # Avoid singularities at leading and trailing edges
    eps = 1e-6
    ii = np.abs(x-0.5)<(0.5-eps)
    s[ii] = (z[ii] - x[ii]*zte) / ( np.sqrt(x[ii]) * (1. - x[ii]) )
    return s

def z_from_shape_space( x, s, zte ):
    """Transform shape space to real coordinates."""
    return np.sqrt(x) * (1. - x) * s + x*zte

def evaluate_coeffs( x, A ):
    n = len(A)-1
    return np.sum(np.stack([A[i] * bernstein( x, n , i ) for i in range(0,n+1)]),0)

def fit_coeffs( x, s, order, dx=0., Rle=None ):
    n = order - 1
    itrim = np.abs(x-0.5)<(0.5-dx)
    xtrim = x[itrim]
    strim = s[itrim]

    X = np.stack( [ bernstein( xtrim, n, i ) for i in range(0,n+1) ] ).T

    if Rle:
        X = X[:,1:]
        A0 = np.sign(Rle)*np.sqrt(2.*np.abs(Rle))
        A, resid = lstsq(X, strim - A0*bernstein( xtrim, n, 0), rcond=None)[:2]
        A = np.insert(A, 0, A0)
        return A, resid
    else:
        return lstsq(X, strim, rcond=None)[:2]


def fit_section( xy, order, zte, dx=0. ):
    # Initially fit with free leading edge radii
    s = [z_to_shape_space(*xyi, zte) for xyi in xy]
    A, _ = zip(*[fit_coeffs( xyi[0,:], si, order, dx ) for xyi, si in zip(xy, s) ])
    A0 = np.array([np.abs(Ai[0]) for Ai in A])
    Rle = 0.5*A0**2.
    guess_Rle = np.mean(Rle)
    err_Rle = np.ptp(Rle)/guess_Rle
    A = [fit_coeffs( xyi[0,:], si, order, dx, guess_Rle*np.sign(xyi[1,2]) )[0] for xyi, si in zip(xy, s) ]
    return A


def resample_coeffs( A, order ):
    x = np.linspace(0.,1.)
    s = evaluate_coeffs( x, A )
    return fit_coeffs( x, s, order )


if __name__=="__main__":


    xy = read_seglid( 'naca6412.dat' )

    zte = 0.00
    order = 4
    dx = 0.05

    A = fit_section( xy, order, zte, dx )


    xfit =  np.linspace(0.,1.,10000)
    sfit = [evaluate_coeffs( xfit, Ai ) for Ai in A]

    yfit = [ z_from_shape_space( xfit, sfiti, zte) for sfiti in sfit]

    # x = np.linspace(0,1.)
    # s = - (x**3 - x**2. + .2*x)
    # A1, _ = fit_coeffs( x, s, 4 )
    # A2, _ = resample_coeffs( A1, 3 )
    # A3, _ = resample_coeffs( A1, 10 )

    f, a = plt.subplots()
    # f2, a2 = plt.subplots()
    for xyi, sfiti, yfiti in zip(xy, sfit, yfit):
        # a2.plot(xyi[0,:],si,'-')
        # a2.plot(xfit,sfiti,'--')
        a.plot(*xyi,'kx')
        a.plot( xfit, yfiti, 'k-')
    # a.plot(*xy2,'kx')
    a.axis('equal')
    # a2.axis('equal')
    plt.show()
