import numpy as np

from scipy.special import gammaln
from copia.data import AbundanceData, CopiaData, IncidenceData


def estimate_coverage(ds: CopiaData, max_steps=None, step_size=1):
    if isinstance(ds, AbundanceData):
        n, f1, f2, T, counts = ds.n, ds.f1, ds.f2, ds.n, ds.counts
    elif isinstance(ds, IncidenceData):
        n, f1, f2, T, counts = ds.n, ds.f1, ds.f2, ds.T, ds.counts
    else:
        raise ValueError("ds Must be an instance of AbundanceData or IncidenceData")
    
    if f2 == 0:
        f0 = (T - 1) / T * f1 * (f1 - 1) / 2
    else:
        f0 = (T - 1) / T * f1**2 / 2 / f2
    A = 1
    if f1 > 0:
        A = T * f0 / (T * f0 + f1)

    def _sub(t):
        if t is None or t == T:
            return 1 - f1 / n * A        
        elif t < T:
            yy = counts[(T - counts) >= t]
            return 1 - sum(yy / n * np.exp(
                gammaln(T - yy + 1) -
                gammaln(T - yy - t + 1) -
                gammaln(T) +
                gammaln(T - t)))
        elif t > T:
            return 1 - f1 / n * A**(t - T + 1)

    return np.array(
        [_sub(t_i) for t_i in np.arange(1, max_steps + step_size, step_size)])


# def estimate_abundance_based_coverage(ds: AbundanceData, m=None):
#     n, f1, f2, counts = ds.n, ds.f1, ds.f2, ds.counts

#     a = 0
#     if f2 > 0:
#         a = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
#     elif f1 > 1:
#         a = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)

#     def _sub(m):
#         # estimation for sample size
#         if m is None or m == n:
#             return 1 - f1 / n * a
#         if m < n:
#             xx = counts[(n - counts) >= m]
#             return 1 - np.sum(xx / n * np.exp(
#                 gammaln(n - xx + 1) -
#                 gammaln(n - xx - m + 1) -
#                 gammaln(n) +
#                 gammaln(n - m)))
#         # extrapolation        
#         if m > n:
#             return 1 - f1 / n * a**(m - n + 1)

#     # if m is None or isinstance(m, int):
#     #     return _sub(m)
    
#     return np.array([_sub(m_i) for m_i in np.arange(1, m + 1)])

# def estimate_incidence_based_coverage(ds: IncidenceData, t=None):
#     n, f1, f2, T, counts = ds.n, ds.f1, ds.f2, ds.T, ds.counts
#     if f2 == 0:
#         f0 = (T - 1) / T * f1 * (f1 - 1) / 2
#     else:
#         f0 = (T - 1) / T * f1**2 / 2 / f2
#     A = 1
#     if f1 > 0:
#         A = T * f0 / (T * f0 + f1)

#     def _sub(t):
#         if t is None or t == T:
#             return 1 - f1 / n * A        
#         elif t < T:
#             yy = counts[(T - counts) >= t]
#             return 1 - sum(yy / n * np.exp(
#                 gammaln(T - yy + 1) -
#                 gammaln(T - yy - t + 1) -
#                 gammaln(T) +
#                 gammaln(T - t)))
#         elif t > T:
#             return 1 - f1 / n * A**(t - T + 1)

#     # if t is None or isinstance(t, int):
#     #     return _sub(t)
    
#     return np.array([_sub(t_i) for t_i in np.arange(1, t + 1)])
