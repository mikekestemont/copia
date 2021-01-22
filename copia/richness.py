# -*- coding: utf-8 -*-
"""
Bias-correcting richness estimators for abundance data
"""

import numpy as np

def chao1(x):
    """
    Estimate bias-corrected species richness in an assemblage.

    Parameters
    ----------
    x : array-like of shape (number of species)
        An array representing the observed abundances (observed
        counts for each individual species. 

    Returns
    -------
    richness : float
        The chao1 estimate ($\hat{f_0}$) of the bias-corrected species richness:

        $\hat{f_0} = \left\{\begin{aligned}
        \frac{(n - 1)}{n} \frac{f_1^2}{(2f_2)} \qquad if f_2 > 0;\\
        \frac{(n - 1)}{n} \frac{f_1(f_1 - 1)}{2} \qquad if f_2 = 0
        \end{aligned}$

        With:
            - $f_1$ = the number of species sighted exactly once in
            the sample (singletons),
            - $f_2$ = the number of species that were sighted twice
            (doubletons)
            - $n$ = the observed, total sample size.
            - $\hat{f_0}$ = the estimated lower bound for the number
            of species that do exist in the assemblage, but which were
            sighted zero times, i.e. the number of undetected species.       

    References
    -------
    - A. Chao, 'Non-parametric estimation of the classes in a population',
    Scandinavian Journal of Statistics (1984), 265-270.
    - A. Chao, et al., 'Quantifying sample completeness and comparing
    diversities among assemblages', Ecological research (2020), 292-314.
    """

    x = x[x > 0]
    n = x.sum()
    t = x.shape[0]
    f1 = (x == 1).sum()
    f2 = (x == 2).sum()

    if f2 > 0:
        return t + (n - 1) / n * (f1 ** 2 / 2 / f2) 
    else:
        return f1 * (f1 - 1) / 2


def egghe_proot():
    pass

def jackknife():
    pass

def min_add_sample():
    pass


def richness(x, method=None, **kwargs):
    x = np.asarray(x, dtype=np.float64)
    if method is None:
        method = "richness"

    estimate = get_estimator(method)(x, **kwargs)
    return estimate
