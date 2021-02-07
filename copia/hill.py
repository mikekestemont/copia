# -*- coding: utf-8 -*-
"""
Hill number calculations (including evenness calculations)
"""
from functools import partial

import numpy as np
from scipy.special import digamma
from scipy.special import binom as choose

import copia.stats as stats

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
    A = np.sum(np.exp(stats.lchoose(x, q) - stats.lchoose(n, q)))
    return np.nan if A == 0 else A**(1 / (1 - q))


def _chao_7d(x, n, f1, p1, q):
    data, counts = np.unique(x, return_counts=True)
    term = np.zeros(data.shape[0])
    zi = stats.lchoose(n, data)
    for i, z in enumerate(data):
        k = np.arange(n - z + 1)
        term[i] = np.sum(
            choose(k - q, k) * np.exp(stats.lchoose(n - k - 1, z - 1) - zi[i]))
    A = np.sum(counts * term)
    if f1 == 0 or p1 == 1:
        B = 0
    else:
        B = f1 / n * (1 - p1)**(1. - n)
        r = np.arange(n)
        B *= (p1**(q - 1)) - np.sum(choose(q - 1, r) * (p1 - 1) ** r)
    return (A + B)**(1 / (1 - q))


def estimated_hill(x, q_values):
    """
    Estimated Hill numbers
    """
    x, n = x[x > 0], x.sum()
    t = x.shape[0]  # number of nonzero traits
    f1 = np.count_nonzero(x == 1)
    f2 = np.count_nonzero(x == 2)

    p1 = 1  # cf equation 6b
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
            return _chao_7b(x, n, f1, p1)
        elif abs(q - round(q)) == 0:
            return _chao_7c(x, q, n)
        else:
            return _chao_7d(x, n, f1, p1, q)

    return np.array([sub(q) for q in q_values])


def empirical_hill(x, q_values):
    """
    Empirical Hill numbers
    """
    p = x[x > 0] / x.sum()

    def sub(q):
        return ((x > 0).sum() if q == 0 else np.exp(-np.sum(p * np.log(p)))
                if q == 1 else np.exp(1 / (1 - q) * np.log(np.sum(p**q))))

    return np.array([sub(q) for q in q_values])


def hill_numbers(x, q_min=0, q_max=3, step=0.1,
                 n_iter=1000, conf=0.95, n_jobs=1, seed=None):
    x = np.array(x, dtype=np.int64)
    q = np.arange(q_min, q_max + step, step)

    emp = stats.bootstrap(x, fn=partial(empirical_hill, q_values=q),
                    n_iter=n_iter,
                    conf=conf,
                    n_jobs=n_jobs,
                    seed=seed)

    est = stats.bootstrap(x, fn=partial(estimated_hill, q_values=q),
                    n_iter=n_iter,
                    conf=conf,
                    n_jobs=n_jobs,
                    seed=seed)
    return emp, est

__all__ = ['estimated_hill', 'empirical_hill', 'hill_numbers']