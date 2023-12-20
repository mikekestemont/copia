import numpy as np
from scipy.special import gammaln
from copia.stats import lchoose


def estimate_incidence_based_coverage(sampling_units, incidence_freqs, t=None):
    incidence_freqs = incidence_freqs[incidence_freqs > 0]
    U = incidence_freqs.sum()
    Q1 = np.count_nonzero(incidence_freqs == 1)
    Q2 = np.count_nonzero(incidence_freqs == 2)
    if Q2 == 0:
        Q0 = (sampling_units - 1) / sampling_units * Q1 * (Q1 - 1) / 2
    else:
        Q0 = (sampling_units - 1) / sampling_units * Q1**2 / 2 / Q2
    A = 1
    if Q1 > 0:
        A = sampling_units * Q0 / (sampling_units * Q0 + Q1)

    def _sub(t):
        if t is None or t == sampling_units:
            return 1 - Q1 / U * A        
        elif t < sampling_units:
            yy = incidence_freqs[(sampling_units - incidence_freqs) >= t]
            return 1 - sum(yy / U * np.exp(
                gammaln(sampling_units - yy + 1) -
                gammaln(sampling_units - yy - t + 1) -
                gammaln(sampling_units) +
                gammaln(sampling_units - t)))
        elif t > sampling_units:
            return 1 - Q1 / U * A**(t - nT + 1)

    if t is None or isinstance(t, int):
        return _sub(t)
    
    return np.array([_sub(t_i) for t_i in t])


def estimate_abundance_based_coverage(x, m=None):
    x, n = x[x > 0], x.sum()
    f1 = np.count_nonzero(x == 1)
    f2 = np.count_nonzero(x == 2)

    a = 0
    if f2 > 0:
        a = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
    elif f1 > 1:
        a = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)

    def _sub(m):
        # estimation for sample size
        if m is None or m == n:
            return 1 - f1 / n * a        
        # interpolation
        if m < n:
            return 1 - np.sum(x / n * np.exp(
                lchoose(n - x, m) - lchoose(n - 1, m)))
        # extrapolation        
        if m > n:
            return 1 - f1 / n * a**(m - n + 1)

    if m is None or isinstance(m, int):
        return _sub(m)
    
    return np.array([_sub(m_i) for i in m])

