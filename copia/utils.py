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
                    dtype=np.int)

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
        s['lci'] = 1 / (d['lci'] / empirical)
        s['uci'] = 1 / (d['uci'] / empirical)
        
    else:
        # normalize to proportions:
        empirical = richness.diversity(assemblage, method='empirical', species=True)
        s['survival'] = empirical / d['richness']
        if 'bootstrap' in d:
            s['bootstrap'] = empirical / d['bootstrap']
        s['lci'] = empirical / d['lci']
        s['uci'] = empirical / d['uci']

    return s


def evenness(d):
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

    Returns
    -------
    evennesses : 1D np.array

        The evenness profile calculated here is:

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
    """
    evenness = (d['richness'] - 1) / (d['richness'][0] - 1)
    return evenness

    """
    # (questionable) hack to obtain a CI:
    lci, uci, richness = d['lci'], d['uci'], d['richness']
    if incl_CI:
            # experimental...
            lci = (lci - 1) / (max(max(lci), lci[0]) - 1)
            uci = (uci - 1) / (max(max(uci), uci[0]) - 1)

            lci = np.maximum(richness, lci)
            uci = np.minimum(richness, uci)
            
            ax.plot(q, lci, c=f"C{i}", linewidth=.8)
            ax.plot(q, uci, c=f"C{i}", linewidth=.8)
            ax.fill_between(q, lci, uci, color=f"C{i}", alpha=0.3)
    """


__all__ = ['to_abundance', 'basic_stats', 'Parallel',
           'check_random_state', 'survival_ratio',
           'evenness', 'bincount']
