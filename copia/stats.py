"""
Miscellaneous statistical subroutines, in particular for
bootstrapping and rarefaction/extrapolation.

Functions based on the R code provided in 
- http://chao.stat.nthu.edu.tw/wordpress/paper/113_Rcode.txt, and
- https://github.com/AnneChao/SpadeR/blob/master/R/Diversity_subroutine.R
"""

import numpy as np
import scipy.stats
from copia.data import AbundanceData

import copia.utils


def lognormal_ci(S_obs, S_est, se, conf=0.95):
    r"""
    Log-normal confidence interval (Chao 1987)

    The number of undetected species is taken to be log-normal, so that the lower
    limit is never below the observed number of species.

    Parameters
    ----------
    S_obs : int
        The observed number of species.
    S_est : float
        The estimated number of species.
    se : float
        The standard error of the estimate.
    conf : float (default = 0.95)
        Confidence level for the confidence interval.

    Returns
    -------
    lci, uci : float
        The limits of the interval.

    References
    ----------
    - A. Chao, 'Estimating the population size for capture-recapture data with
      unequal catchability', Biometrics (1987), 783-791 (eq. 12).
    """
    f0 = S_est - S_obs
    if f0 <= 0:
        return S_est, S_est
    z = scipy.stats.norm.ppf((1 + conf) / 2)
    K = np.exp(z * np.sqrt(np.log(1 + se ** 2 / f0 ** 2)))
    return S_obs + f0 / K, S_obs + f0 * K


def dbinom(x, size, prob):
    d = scipy.stats.binom(size, prob).pmf(x)
    return 1 if np.isnan(d) else d


def lchoose(n, k):
    return np.log(scipy.special.comb(n, k))
    # return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def quantile(x, q, weights=None):
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()



