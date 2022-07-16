from dataclasses import dataclass

import numpy as np


@dataclass
class QuadRule:
    pts: np.ndarray
    wts: np.ndarray


def simpson_rule(n, a=-1, b=1):
    """
    Output the points and weights for a Simpson rule quadrature on the interval
    (a, b)
    """
    if not (n >= 3 and n % 2 == 1):
        raise ValueError("Simpson's rule is only defined for odd n >= 3.")
    h = (b - a) / (n - 1)
    pts = np.linspace(a, b, n)
    wts = np.empty(n)
    wts[0] = 1
    wts[1::2] = 4
    wts[2::2] = 2
    wts[-1] = 1
    wts *= h / 3
    return QuadRule(pts, wts)


def composite_rule(q_rule_fnc, *domains):
    pts = []
    wts = []
    for d in domains:
        qr = q_rule_fnc(*d)
        pts.append(qr.pts)
        wts.append(qr.wts)
    pts = np.concatenate(pts)
    wts = np.concatenate(wts)
    return QuadRule(pts, wts)


def gauss_rule(n, a=-1, b=1):
    """
    Points and weights for a Gaussian quadrature with n points on the interval
    (a, b)
    """
    pts, wts = np.polynomial.legendre.leggauss(n)
    pts = (pts + 1) * (b - a) / 2 + a
    wts = wts * (b - a) / 2
    return QuadRule(pts, wts)


def gauss_herm_rule(n, center=0, scale=1.0):
    """Points and weights for the Gauss-Hermite quadrature rule with n points,
    scaled so that the weight function is exp(-((x - center) / scale)^2).

    Args:
        n: Number of quadrature points
        center: Weight function center. Defaults to 0.
        scale: Weight function scaling. Defaults to 1.0.
    """
    orig_pts, orig_wts = np.polynomial.hermite.hermgauss(n)
    orig_wts /= np.exp(-(orig_pts**2))
    pts = (orig_pts - center) * scale
    wts = orig_wts * scale
    return QuadRule(pts, wts)


def log_gauss_rule(N: int, a: float, b: float):
    """Return a Gaussian quadrature rule in the log domain (log(a), log(b))

    Args:
        N: The number of quadrature points
        a: Left end of interval
        b: Right end of interval

    Returns:
        QuadRule: The points and weights.
    """
    A = np.log(a)
    B = np.log(b)
    qr = gauss_rule(N, a=A, b=B)
    pts = np.exp(qr.pts)
    wts = np.exp(qr.pts) * qr.wts
    return QuadRule(pts, wts)
