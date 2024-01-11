"""
Hill number calculations (including evenness calculations)
"""
from functools import partial

import numpy as np
from scipy.special import digamma
from scipy.special import binom as choose

from copia.bootstrap import bootstrap_abundance_data
from copia.stats import lchoose
from copia.data import AbundanceData


def _chao_7a(t, n, f1, f2):
    return (t + (n - 1) / n * ((f1**2 / 2 / f2) if f2 > 0 else
                               (f1 * (f1 - 1) / 2)))


def _chao_7b(x, n, f1, p1):
    A = np.sum(x / n * (digamma(n) - digamma(x)))
    if f1 == 0 or p1 == 1:
        B = 0
    else:
        B = f1 / n * (1 - p1)**(1. - n)
        r = np.arange(1, n)
        B *= (-np.log(p1) - np.sum((1 - p1)**r / r))
    return np.exp(A + B)


def _chao_7c(x, q, n):
    A = np.sum(np.exp(lchoose(x, q) - lchoose(n, q)))
    return np.nan if A == 0 else A**(1 / (1 - q))


def _chao_7d(x, n, f1, p1, q):
    data, counts = np.unique(x, return_counts=True)
    term = np.zeros(data.shape[0])
    zi = lchoose(n, data)
    for i, z in enumerate(data):
        k = np.arange(n - z + 1)
        term[i] = np.sum(
            choose(k - q, k) * np.exp(lchoose(n - k - 1, z - 1) - zi[i]))
    A = np.sum(counts * term)
    if f1 == 0 or p1 == 1:
        B = 0
    else:
        B = f1 / n * (1 - p1)**(1. - n)
        r = np.arange(n)
        B *= (p1**(q - 1)) - np.sum(choose(q - 1, r) * (p1 - 1) ** r)
    return (A + B)**(1 / (1 - q))


def compute_true_hill_numbers(ds: AbundanceData, q_values):
    """
    Estimated Hill numbers
    """

    p1 = 1  # cf equation 6b
    t, n, f1, f2 = ds.S_obs, ds.n, ds.f1, ds.f2
    counts = ds.counts
    if f2 > 0:
        p1 = 2 * f2 / ((n - 1) * f1 + 2 * f2)
    elif f1 > 0:
        p1 = 2 / ((n - 1) * (f1 - 1) + 2)

    def sub(q):
        # equation 7a
        if q == 0:
            return _chao_7a(t, n, f1, f2)
        # equation 7b
        elif q == 1:
            return _chao_7b(counts, n, f1, p1)
        elif abs(q - round(q)) == 0:
            return _chao_7c(counts, q, n)
        else:
            return _chao_7d(counts, n, f1, p1, q)

    return np.array([sub(q) for q in q_values])


def compute_empirical_hill_numbers(ds: AbundanceData, q_values):
    """
    Empirical Hill numbers
    """
    x = ds.counts
    p = x[x > 0] / x.sum()

    def sub(q):
        return ((x > 0).sum() if q == 0 else np.exp(-np.sum(p * np.log(p)))
                if q == 1 else np.exp(1 / (1 - q) * np.log(np.sum(p**q))))

    return np.array([sub(q) for q in q_values])


def compute_hill_numbers(ds: AbundanceData, q_min=0, q_max=3, steps=100,
                         estimate_unseen=False, CI=False, n_iter=1000,
                         conf=0.89, n_jobs=1, seed=None):
    q = np.linspace(q_min, q_max, num=steps)
    if not estimate_unseen:
        hill_function = compute_empirical_hill_numbers
    else:
        hill_function = compute_true_hill_numbers

    if not CI:
        hill_numbers = hill_function(ds, q)
    else:
        hill_numbers = bootstrap_abundance_data(
                ds, fn=partial(hill_function, q_values=q),
                n_iter=n_iter,
                conf=conf,
                n_jobs=n_jobs,
                seed=seed)
    return hill_numbers

# TODO: EIGENLIJK EEN TRANSFORMATIE EN GEEN ESTIMATIE
def compute_evenness(d: dict, q_min=0, q_max=3, step=0.1, E=3):
    r"""
    Evenness: calculation of a (normalized) evenness profile

    Parameters
    ----------
    d : dict
        A `dict`, minimally containing a 'richness' key that
        indexes :math:`{}^qD`: a 1D np.array with the Hill number
        profile for a certain range of orders (:math:`q`). This
        array can represent an estimated (bias-corrected) or an
        empirical (observed) Hill profile.
    q_min : float (default = 0)
        Minimum order to consider. Only relevant when CV is True.
    q_max : float (default = 3)
        Maximum order to consider. Only relevant when CV is True.
    step : float (default = 0.1)
        Step size in between consecutive orders. Only relevant
        when CV is True.
    CV : bool (default = False)
        If this flag is set to `True`, another class of evenness
        measures is being calculated, based on the coefficient of
        variation or CV (see below).

    Returns
    -------
    evennesses : 1D np.array

        The (default) evenness profile calculated here is:

        .. math::
            ({}^qD - 1) / (\hat{S} - 1)

        With:
            - :math:`{}^qD` = the Hill number profile passed
            - :math:`\hat{S}` = the number of distinct species at
              :math:`q=0` (here equated to the first value encountered
              in (:math:`{}^qD`))
        
        The resulting profile will be normalized (bounded to the range
        [0, 1]), enabling a direct comparison between assemblages of 
        different sizes.

        When `CV` is explicitly set to `True`, another class of evenness
        measures is being calculated, based on the coefficient of
        variation:

        .. math::
            {}^qE^* = [1-(^qD)^{1-q}] / (1-\hat{S}^{1-q})

        When $q$ tends to 1, this reduces to the Shannon-entropy divided by
        $log(S)$, which is known as Pielou’s $J'$ evenness index (Pielou
        1966); for $q=2$, the corresponding evenness measure is:
        $1 - (CV)^2/S$. 


    Note
    ----
        The recommended usage of this function is to apply it to one of
        the dicts returned by `copia.diversity.hill_numbers()`.

    References
    ----------
    - A. Chao and R. Carlo, 'Quantifying evenness and linking it to
      diversity, beta diversity, and similarity', Ecology (2019),
      e02852.
    - A. Chao, et al., 'Quantifying sample completeness and comparing
      diversities among assemblages', Ecological research (2020),
      292-314.
    - E.C. Pielou, 'The measurement of diversity in different types
      of biological collections. Journal of Theoretical Biology (1966),
      131-144.
    """
    qs = np.arange(q_min, q_max + step, step)
            
    if E in (1, 2):
        qs = (1 - qs) if E == 1 else (qs - 1)
        evenness = ((1 - (d['richness'] ** qs)) /
                    (1 - (d['richness'][0] ** qs)))
        # for q = 1, the corresponding evenness measure is  1- CV^2/S
        i = int(1 / step)
        evenness[i] = np.log(d['richness'][i]) / np.log(d['richness'][0])
    elif E == 3:
        evenness = (d['richness'] - 1) / (d['richness'][0] - 1)
    elif E == 4:
        evenness = (1 - 1 / d['richness']) / (1 - 1 / d['richness'][0])
    elif E == 5:
        evenness = np.log(d['richness']) / np.log(d['richness'][0])
    else:
        raise ValueError("Evenness type not implemented")
    return evenness


