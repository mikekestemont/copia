# -*- coding: utf-8 -*-
"""
Bias-correcting richness estimators for abundance data
"""

import numpy as np

def chao1(x):
    """
    Chao1 estimate of bias-corrected species richness

    Parameters
    ----------
    x : array-like of shape (number of species)
        An array representing the observed abundances (observed
        counts for each individual species. 

    Returns
    -------
    richness : float
        Estimate ($\hat{f_0}$) of the bias-corrected species richness:

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

    x = np.array(x, dtype=np.float64)

    x = x[x > 0]
    n = x.sum()
    t = x.shape[0]
    f1 = (x == 1).sum()
    f2 = (x == 2).sum()

    if f2 > 0:
        return t + (n - 1) / n * (f1 ** 2 / 2 / f2) 
    else:
        return f1 * (f1 - 1) / 2


def egghe_proot(x, alpha=150):
    """
    Egghe & Proot estimate of bias-corrected species richness

    Parameters
    ----------
    x : array-like of shape (number of species)
        An array representing the observed abundances (observed
        counts for each individual species.
    alpha : float
        An estimate of the average print run

    Returns
    -------
    richness : float
        Estimate ($\hat{f_0}$) of the bias-corrected species richness:

        $\hat{f_0} = \left( \frac{1}{1 + \frac{2f_2}{(a-1)f_1}} \right)^a$

        With:
            - $f_1$ = the number of species sighted exactly once in
            the sample (singletons),
            - $f_2$ = the number of species that were sighted twice
            (doubletons)
            - $\hat{f_0}$ = the estimated number of species that once 
            existed in the assemblage, but which were sighted zero times, 
            i.e. the number of undetected species.

    References
    -------
    - L. Egghe and G. Proot, 'The estimation of the number of lost
    multi-copy documents: A new type of informetrics theory', Journal
    of Informetrics (2007), 257-268.
    - Q.L. Burrell, 'Some comments on "The estimation of lost multi-copy
    documents: A new type of informetrics theory" by Egghe and Proot',
    Journal of Informetrics (2008), 101–105.
    """


    x = np.array(x, dtype=np.int64)

    ft = np.bincount(x)[1:]
    S = ft.sum()

    P1 = (x == 1).sum()
    P2 = (x == 2).sum()
    P0 = (1 / (1 + (2 / (alpha - 1)) * (P2 / P1))) ** alpha

    S_lost = S * (P0 / (1 - P0))
    S_lost = S + S_lost

    if not np.isinf(S_lost):
        return S_lost
    else:
        return np.nan


def jackknife():
    pass

def min_add_sample():
    pass

