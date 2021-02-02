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


def basic_stats(x):
    assert isinstance(x, np.ndarray)
    return {'f1': np.count_nonzero(x == 1),
            'f2': np.count_nonzero(x == 2),
            'f3': np.count_nonzero(x == 3),
            'f4': np.count_nonzero(x == 4),
            'S': (x > 0).sum(),
            'n': x.sum()}


class Parallel:
    def __init__(self, n_workers, n_tasks):
        self.pool = mp.Pool(n_workers)
        self._results = []
        self._pb = tqdm.tqdm(total=n_tasks)

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
    evenness = (d['richness'] - 1) / (d['richness'][0] - 1)
    return evenness

    """
    hack for the CI
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


