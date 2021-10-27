# -*- coding: utf-8 -*-
"""
Various utility functions.
"""
from collections import Counter
import multiprocessing as mp

import numpy as np
import tqdm

import copia.richness as richness


def to_abundance(species):
    return np.array(tuple(Counter(species).values()),
                    dtype=np.int64)

def bincount(x):
    _, x = np.unique(x.flatten(), return_inverse=True)
    return np.bincount(x)


def basic_stats(x):
    assert isinstance(x, np.ndarray)
    return {'f1': np.count_nonzero(x == 1),
            'f2': np.count_nonzero(x == 2),
            'f3': np.count_nonzero(x == 3),
            'f4': np.count_nonzero(x == 4),
            'S': (x > 0).sum(),
            'n': x.sum()}


class Parallel:
    r"""
    Helper class for parallel execution.
    """
    def __init__(self, n_workers, n_tasks, disable_pb=False):
        self.pool = mp.Pool(n_workers)
        self._results = []
        self._pb = tqdm.tqdm(total=n_tasks, disable=disable_pb)

    def apply_async(self, fn, args=None):
        self.pool.apply_async(fn, args=args, callback=self._completed)

    def _completed(self, result):
        self._results.append(result)
        self._pb.update()

    def join(self):
        self.pool.close()
        self.pool.join()

    def result(self):
        self._pb.close()
        self.pool.close()
        return self._results


def check_random_state(seed):
    r"""
    Helper class to manage stable random
    number generators.
    """
    if seed is np.random:
        return np.random.mtrand._rand
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, int):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


def survival_ratio(assemblage, method='chao1', **kwargs):
    r"""
    Calculates the survival ratio of an assemblage

    Parameters
    ----------
    assemblage : 1D numpy array with shape (number of species)
        An array representing the abundances (observed
        counts) for each individual species.
    method : str (default = "chao1")
        The diversity estimator to apply (with CI set to true)
    **kwargs : additional arguments passed to the estimator

    Returns
    -------
    s : dict
        The returned dict `s` will have the following fields:
            - "survival" (float) = the unbiased survival estimate
            - "lci" (float) = lower confidence interval
            - "uci" (float) = upper confidence interval
            - "bootstrap" (1D np.array) = bootstrap values obtained
              for the survival estimate.
        
        In ecological terms, we calculate the survival ratio
        as the sample completeness at order 0. For species
        diversity, this estimate can be obtained as:
        :math:`S_{obs}/ \hat{S}`. For minsample, we estimate
        the survival ratio as: :math:`n / n + m`.

        With:
            - :math:`S_{obs}` = the observed diversity
            - :math:`\hat{S}` = the bias-corrected diversity
            - :math:`n` = the observed population size.
            - :math:`m` = the estimated number of additional
              samples required.

    References
    ----------
    - A. Chao, et al., 'Quantifying sample completeness and comparing
      diversities among assemblages', Ecological research (2020),
      292-314.
    - M. Kestemont & F. Karsdorp, 'Estimating the Loss of Medieval
      Literature with an Unseen Species Model from Ecodiversity',
      Computational Humanities Research (2020), 44-55.
    """
    method = method.lower()
    
    d = richness.diversity(assemblage, method=method, CI=True, **kwargs)
    s = {}
        
    if method == 'minsample':
        # normalize to proportions:
        empirical = richness.diversity(assemblage, method='empirical', species=False)
        s['survival'] = 1 / (d['richness'] / empirical)
        if 'bootstrap' in d:
            s['bootstrap'] = 1 / (d['bootstrap'] / empirical)
        # note: upper and lower CI have to be swapped
        s['lci'] = 1 / (d['uci'] / empirical)
        s['uci'] = 1 / (d['lci'] / empirical)
        
    else:
        # normalize to proportions:
        empirical = richness.diversity(assemblage, method='empirical', species=True)
        s['survival'] = empirical / d['richness']
        if 'bootstrap' in d:
            s['bootstrap'] = empirical / d['bootstrap']
        # note: upper and lower CI have to be swapped
        s['lci'] = empirical / d['uci']
        s['uci'] = empirical / d['lci']

    return s


def evenness(d, q_min=0, q_max=3, step=0.1, E=3):
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
        the dicts returned by `copia.hill.hill_numbers()`.

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
    return evenness

__all__ = ['to_abundance', 'basic_stats', 'Parallel',
           'check_random_state', 'survival_ratio',
           'evenness', 'bincount']
