"""Some test stuff for shape-space."""
import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from scipy.optimize import newton, fmin


def read_seglid(file_name):
    """Load a seglid data file with aerofoil section geometry."""
    with open(file_name, "r") as f:
        # Skip header
        lines = f.readlines()[1:]
    # Parse lines into x and y coordinates
    xy = np.array([[float(i) for i in line.strip().split()] for line in lines]).T
    # Split into pressure and suction sides by turning point of x-coord
    i1 = np.where(np.diff(xy[0, :]) > 0.0)[0][0]
    xy1 = np.flip(xy[:, : (i1 + 1)], 1)
    xy2 = xy[:, i1:-1]
    # Determine axial chord and normalise
    xle = np.mean((xy1[0, 0], xy2[0, 0]))
    xte = np.mean((xy1[0, -1], xy2[0, -1]))
    c = xte - xle
    xy1 /= c
    xy2 /= c
    return xy1, xy2


def bernstein(x, n, i):
    """Evaluate ith Bernstein polynomial of order n at some x-coordinates."""
    return binom(n, i) * x ** i * (1.0 - x) ** (n - i)


def to_shape_space(x, z, zte):
    """Transform real thickness to shape space."""
    # Avoid singularities at leading and trailing edges
    eps = 1e-6
    ii = np.abs(x - 0.5) < (0.5 - eps)
    s = np.ones(x.shape) * np.nan
    s[ii] = (z[ii] - x[ii] * zte) / (np.sqrt(x[ii]) * (1.0 - x[ii]))
    return s


def from_shape_space(x, s, zte):
    """Transform shape space to real coordinates."""
    return np.sqrt(x) * (1.0 - x) * s + x * zte


def evaluate_coefficients(x, A):
    """Evaluate a set of Bernstein polynomial coefficients at some x-coords."""
    n = len(A) - 1
    return np.sum(
        np.stack([A[i] * bernstein(x, n, i) for i in range(0, n + 1)]), axis=0
    )


def fit_coefficients(x, s, order, dx=0.05):
    """Fit shape-space distribution with Bernstein polynomial coefficients.

    Return both a vector of coefficients, length `order`, and sum residual."""
    n = order - 1
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    itrim = np.abs(x - 0.5) < (0.5 - dx)
    xtrim = x[itrim]
    strim = s[itrim]
    # Evaluate all polynomials
    X = np.stack([bernstein(xtrim, n, i) for i in range(0, n + 1)]).T
    # All coefficients free - normal fit
    return lstsq(X, strim, rcond=None)[:2]

def fit_both(xy, chi, zte, order):
    """Fit Bernstein polynomial coefficients for both aerofoils surfaces."""
    n = order - 1
    dx = 0.02
    # When converting from real coordinates to shape space, we end up with
    # singularities and numerical instability at leading and trailing edges.
    # So in these cases, ignore within dx at LE and TE
    xtrim_all = []
    strim_all = []
    X_all = []
    X_le_all = []
    for xyi in xy:
        xc, yc, t = coord_to_thickness(xyi, chi)
        s = to_shape_space(xc, t, zte)
        itrim = np.abs(xc - 0.5) < (0.5 - dx)
        xtrim_all.append(xc[itrim])
        strim_all.append(s[itrim])
        X_all.append(np.stack([bernstein(xc[itrim], n, i) for i in range(1, n + 1)]).T)
        X_le_all.append(bernstein(xc[itrim], n, 0))

    strim = np.concatenate(strim_all)
    X_le = np.concatenate(X_le_all)
    X = np.block([[X_all[0],np.zeros(X_all[1].shape)],[np.zeros(X_all[0].shape),X_all[1]]])
    X = np.insert(X, 0, X_le, 1)
    A_all, resid = lstsq(X, strim, rcond=None)[:2]
    Au = A_all[:order]
    Al = np.insert(A_all[order:],0, A_all[0])
    return (Au, Al), resid

def resample_coefficients(A, order):
    """Up- or down-sample a set of coefficients to a new order."""
    x = np.linspace(0.0, 1.0)
    s = evaluate_coefficients(x, A)
    return fit_coefficients(x, s, order)


def evaluate_camber(x, chi):
    """Camber line as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    return tanchi[0] * x + 0.5 * (tanchi[1] - tanchi[0]) * x ** 2.0


def evaluate_camber_slope(x, chi):
    """Camber line slope as a function of x, given inlet and exit angles."""
    tanchi = np.tan(np.radians(chi))
    return tanchi[0] + (tanchi[1] - tanchi[0]) * x


def coord_to_thickness(xy, chi):
    """Perpendicular thickness distribution given camber line angles."""
    xu, yu = xy
    tanchi = np.tan(np.radians(chi))
    # Find intersections of xu, yu with camber line perpendicular
    def iterate(x):
        return yu - evaluate_camber(x, chi) + (xu - x) / evaluate_camber_slope(x, chi)
    xc = newton(iterate, xu)
    yc = evaluate_camber(xc, chi)
    # Now evaluate thickness
    t = np.sqrt(np.sum(np.stack((xu - xc, yu - yc)) ** 2.0, 0))
    return xc, yc, t


def thickness_to_coord(xc, t, chi):
    theta = np.arctan(evaluate_camber_slope(xc, chi))
    yc = evaluate_camber(xc, chi)
    xu = xc - t * np.sin(theta)
    yu = yc + t * np.cos(theta)
    return xu, yu


def fit_surface(xy, chi, zte, order, A0=None):
    """Fit coefficients for perpendicular thickness in shape space."""
    # Convert to perpendicular thickness
    xc, yc, t = coord_to_thickness(xy, chi)
    # Transform thickness to shape-space
    s = to_shape_space(xc, t, zte)
    # Fit in shape space
    return fit_coefficients(xc, s, order, dx=0.02, A0=A0)


def evaluate_surface(A, x, chi, zte):
    """Given a set of coefficients, return coordinates."""
    s = evaluate_coefficients(x, A)
    t = from_shape_space(x, s, zte)
    return thickness_to_coord(x, t, chi)


def fit_aerofoil(xy, chi, zte, order):
    """Fit two sets of coefficients for pressure/suction sides of aerofoil."""
    # First pass, both surfaces with free LE radius
    A = [fit_surface(xyi, chi, zte, order)[0] for xyi in xy]
    # Now fit again with the mean A0
    A0 = np.mean([Ai[0] for Ai in A])
    return zip(*[fit_surface(xyi, chi, zte, order, A0) for xyi in xy])


def fit_camber(xy, zte, order, chi0):
    def iterate(x):
        _, resid = fit_aerofoil(xy, x, zte, order)
        resid = np.mean(resid)
        return resid

    return fmin(iterate, chi0)


if __name__ == "__main__":

    # Read some input data
    xy = read_seglid("naca6412.dat")

    print(xy[0].size)
    print(xy[1].size)
    quit()

    # Camber line
    chi = (15.0, -15.0)
    xfit = np.linspace(0.,1.)

    # # Thickness plot
    # f, a = plt.subplots()
    # a.axis("equal")
    # # a.axis("off")
    # a.grid(False)
    # # Camber line
    # a.plot(xfit, evaluate_camber(xfit, chi),'k--')
    # for i in range(len(xy)):
    #     # Base coordinates
    #     a.plot(*xy[i], "x")
    #     # Thickness vectors
    #     xc, yc, t = coord_to_thickness(xy[i], chi)
    #     dx = np.stack((xc,xy[i][0,:]))
    #     dy = np.stack((yc,xy[i][1,:]))
    #     a.plot(dx, dy, color='C%d'%i)
    # a.set_xlabel('Axial Coordinate')
    # a.set_ylabel('Tangential Coordinate')
    # plt.tight_layout()
    # plt.savefig("thickness.svg")
    # quit()

    # Thickness/shape space plot
    zte = 0.001
    order = 4
    dx = 0.05

    # xc, yc, t = coord_to_thickness(xy[0], chi)
    # f, a = plt.subplots()
    # s = to_shape_space( xc, t, zte)
    # A, _ = fit_coefficients( xc, s, order )
    # tfit = evaluate_coefficients( xfit, A)
    # a.plot(xc, t,label='$t(x)$')
    # a.plot(xc, s,label='$s(x)$')
    # a.plot(xfit, tfit, '--', label='Fit $n=3$')
    # a.set_ylim([0.,0.2])
    # a.set_xlabel('Axial Coordinate')
    # a.set_ylabel('Thickness')
    # a.legend()
    # plt.tight_layout()
    # plt.savefig("shape-space.svg")
    # # plt.show()
    # quit()

    xfit = np.linspace(0.0, 1.0, 10000)


    # A, resid = fit_aerofoil(xy, chi, zte, order)
    A, residul = fit_both(xy, chi, zte, order)

    f, a = plt.subplots()
    a.axis("equal")
    a.plot(xfit, evaluate_camber(xfit, chi), "k--")
    for i in range(len(xy)):
        xufit, yufit = evaluate_surface(A[i] * (1 - 2 * i), xfit, chi, zte)
        a.plot(*xy[i], "x")
        a.plot(xufit, yufit, "-", color="C%d" % i)
    a.set_xlabel('Axial Coordinate')
    a.set_ylabel('Tangential Coordinate')
    plt.tight_layout()
    plt.savefig("fitted.svg")
    plt.show()
