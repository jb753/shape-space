"""Some test stuff for shape-space."""
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.optimize import newton

def read_seglid(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()[1:]
    # Skip header
    xy = np.array([[float(i) for i in line.strip().split()] for line in lines]).T
    # Find the turning point of x coordinate
    i1 = np.where(np.diff(xy[0,:])>0.)[0][0]
    # Split into pressure and suction sides, normalise
    xy1 = np.flip(xy[:,:(i1+1)],1)
    xy2 = xy[:,i1:-1]
    # Determine axial chord and normalise
    xle = np.mean((xy1[0,0],xy2[0,0]))
    xte = np.mean((xy1[0,-1],xy2[0,-1]))
    c = xte-xle
    xy1 /= c
    xy2 /= c
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

# def fit_section( xy, order, zte, dx=0. ):
#     # Initially fit with free leading edge radii
#     s = [z_to_shape_space(*xyi, zte) for xyi in xy]
#     A, _ = zip(*[fit_coeffs( xyi[0,:], si, order, dx ) for xyi, si in zip(xy, s) ])
#     A0 = np.array([np.abs(Ai[0]) for Ai in A])
#     Rle = 0.5*A0**2.
#     guess_Rle = np.mean(Rle)
#     err_Rle = np.ptp(Rle)/guess_Rle
#     A = [fit_coeffs( xyi[0,:], si, order, dx, guess_Rle*np.sign(xyi[1,2]) )[0] for xyi, si in zip(xy, s) ]
#     return A

def resample_coeffs( A, order ):
    x = np.linspace(0.,1.)
    s = evaluate_coeffs( x, A )
    return fit_coeffs( x, s, order )

def eval_camber( x, chi ):
    tanchi = np.tan(np.radians(chi))
    return x * tanchi[0] + 0.5*(tanchi[1]-tanchi[0])*x*x

def eval_camber_slope( x, chi ):
    tanchi = np.tan(np.radians(chi))
    return tanchi[0] + (tanchi[1]-tanchi[0])*x

def coord_to_thickness( xy, chi ):
    """Perpendicular thickness distribution given camber line angles."""
    xu, yu = xy
    dyu_dx = np.gradient(yu, xu)
    tanchi = np.tan(np.radians(chi))
    # Find intersections of xu, yu with camber line perpendicular
    def iterate( x ):
        return yu - eval_camber( x, chi ) + (xu - x)/eval_camber_slope(x, chi)
    xc = newton( iterate, xu )
    yc = eval_camber( xc, chi )
    return xc, yc


if __name__=="__main__":


    xy = read_seglid( 'naca6412.dat' )

    zte = 0.00
    order = 4
    dx = 0.05
    chi = (15.,-15.)

    xc1, yc1 = coord_to_thickness( xy[0], chi)

    # s = [z_to_shape_space(*xyi, zte) for xyi in xy]
    # A, resid = zip(*[fit_coeffs( xyi[0,:], si, order, dx ) for xyi, si in zip(xy, s) ])
    xfit =  np.linspace(0.,1.,10000)
    # sfit = [evaluate_coeffs( xfit, Ai ) for Ai in A]
    # yfit = [ z_from_shape_space( xfit, sfiti, zte) for sfiti in sfit]

    # x = np.linspace(0,1.)
    # s = - (x**3 - x**2. + .2*x)
    # A1, _ = fit_coeffs( x, s, 4 )
    # A2, _ = resample_coeffs( A1, 3 )
    # A3, _ = resample_coeffs( A1, 10 )

    f, a = plt.subplots()
    # f2, a2 = plt.subplots()
    for xyi in xy:
        # a2.plot(xyi[0,:],si,'-')
        # a2.plot(xfit,sfiti,'--')
        a.plot(*xyi,'x')
    a.plot( xfit, eval_camber(xfit, chi), 'k--')
    a.plot( xc1, yc1, 'o')
    # a.plot(*xy2,'kx')
    a.axis('equal')
    # a2.axis('equal')
    plt.show()
