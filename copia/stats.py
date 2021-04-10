# -*- coding: utf-8 -*-
"""
Miscellaneous statistical subroutines, in particular for
bootstrapping and rarefaction/extrapolation.

Functions based on the R code provided in 
- http://chao.stat.nthu.edu.tw/wordpress/paper/113_Rcode.txt, and
- https://github.com/AnneChao/SpadeR/blob/master/R/Diversity_subroutine.R
"""
import numpy as np
import scipy.stats
from scipy.special import gammaln
from tqdm import tqdm

import copia.utils as utils


def dbinom(x, size, prob):
    d = scipy.stats.binom(size, prob).pmf(x)
    return 1 if np.isnan(d) else d


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def bt_prob(x):
    x, n = x[x > 0], x.sum()
    f1, f2 = (x == 1).sum(), (x == 2).sum()
    C = 1 - f1 / n * (((n - 1) * f1 / ((n - 1) * f1 + 2 * f2)) if f2 > 0 else
                      ((n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)) if f1 > 0 else
                      0)
    W = (1 - C) / np.sum(x / n * (1 - x / n) ** n)
    p = x / n * (1 - W * (1 - x / n) ** n)
    f0 = np.ceil(((n - 1) / n * f1 ** 2 / (2 * f2)) if f2 > 0 else
                 ((n - 1) / n * f1 * (f1 - 1) / 2))
    p0 = (1 - C) / f0
    p = np.hstack((p, np.array([p0 for i in np.arange(f0)])))
    return p


def bootstrap(x, fn,
              n_iter=1000,
              conf=0.95,
              n_jobs=1,
              disable_pb=False,
              seed=None):
    """Bootstrap method to construct confidence intervals of a specified 
    richness index.

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    fn : Callable representing the target richness index.
    n_iter : int (default = 1000)
        Number of bootstrap samples.
    conf : float (default = 0.95)
        Compute the confidence interval at the specified level.
    n_jobs : int (default = 1)
        Number of cores to use for computation.
    seed : int (default = None)
        A seed to initialize the random number generator. 

    Returns
    -------
    estimates : dict
        A dictionary providing the empirical richness index keyed with 
        `richness`, the bootstrapped estimates `bootstrap`, the lower and 
        upper endpoint of the specified confidence interval (`lci` and `uci`), 
        and the standard deviation of the richness index. 
    """
    rnd = utils.check_random_state(seed)
    pro = fn(x) 
    p, n = bt_prob(x), x.sum()
    data_bt = rnd.multinomial(n, p, n_iter)
    
    pool = utils.Parallel(n_jobs, n_iter, disable_pb=disable_pb)
    for row in data_bt:
        pool.apply_async(fn, args=(row,))
    pool.join()

    bt_pro = np.array(pool.result())
    pro_mean = bt_pro.mean(0)
    
    lci_pro = -np.quantile(bt_pro, (1 - conf) / 2, axis=0) + pro_mean
    uci_pro = np.quantile(bt_pro, 1 - (1 - conf) / 2, axis=0) - pro_mean
    sd_pro = np.std(bt_pro, axis=0)

    bt_pro = pro_mean - bt_pro

    lci_pro, uci_pro = pro - lci_pro, pro + uci_pro
    bt_pro = pro - bt_pro

    return {'richness': pro,
            'lci': lci_pro,
            'uci': uci_pro,
            'std': sd_pro,
            'bootstrap': bt_pro}


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


def rarefaction_extrapolation(x, max_steps):
    r"""
    Species accumulation curve (calculation)

    Parameters
    ----------
    x : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    max_steps : int
        Maximum number of steps in the accumulation. Should
        be a positive integer, but can be smaller (rarefaction)
        or larger (extrapolation) than the empirical population
        size (:math:`n`).

    Returns
    -------
    accumulation : np.ndarray
        Species accumulation curve as a 1D numpy array of shape
        (max_steps, ). Contains the estimated richness of the
        assemblage, for each step in the range [0, max_steps].

    References
    ----------
    - N.J. Gotelli and R.K. Colwell, 'Estimating Species Richness',
      Biological Diversity: Frontiers in Measurement and Assessment,
      OUP (2011), 39-54.
    - A. Chao, et al. 'Rarefaction and extrapolation with Hill numbers:
      a framework for sampling and estimation in species diversity studies',
      Ecological Monographs (2014), 84, 45–67.
    """
    x, n = x[x > 0], x.sum()
    def _sub(m):
        if m <= n:
            return np.sum(1 - np.array(
                [np.exp(gammaln(n - i + 1) + gammaln(n - m + 1) - 
                        gammaln(n - i - m + 1) - gammaln(n + 1)) if i <= (n - m) else
                 0 for i in x]))
        else:
            S = (x > 0).sum()
            f1, f2 = (x == 1).sum(), (x == 2).sum()
            f0 = ((n - 1) / n * f1 * (f1 - 1) / 2) if f2 == 0 else ((n - 1) / n * f1**2 / 2 / f2)
            A = n * f0 / (n * f0 + f1)
            return S if f1 == 0 else (S + f0 * (1 - A**(m - n)))
    return np.array([_sub(mi) for mi in range(1, max_steps)])


__all__ = ['rarefaction_extrapolation', 'quantile', 'bootstrap']
