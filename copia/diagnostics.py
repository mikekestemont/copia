"""
Diagnostics for the assumptions of the estimators
"""
import numpy as np
import scipy.stats

from copia.data import AbundanceData, IncidenceData, accepts


@accepts(AbundanceData, IncidenceData)
def one_inflation(ds: AbundanceData, max_count=12):
    r"""
    Test for one-inflation: are there more singletons than the species that were
    observed more often lead one to expect?

    An excess of singletons (e.g. from misidentifications, or because being observed
    makes a species less likely to be observed again) inflates the estimates that
    rest on :math:`f_1`, such as Chao1.

    Parameters
    ----------
    ds : AbundanceData or IncidenceData
        The observed counts for each individual species.
    max_count : int (default = 12)
        The largest count to use. The ratios of the larger counts are noisy.

    Returns
    -------
    test : dict
        With the keys:
            - "f1", "f1_expected" : the observed number of singletons, and the
              number expected from the larger counts.
            - "r1", "r1_expected" : the observed and expected ratio :math:`r_1`.
            - "excess" : the difference between the logarithms of the two. It is
              negative if there are too many singletons.
            - "z" : the same, divided by its standard error.
            - "p" : the p-value of the one-sided test for one-inflation.
            - "aic0", "aic1", "bic0", "bic1" : the information criteria of the
              regressions without (0) and with (1) one-inflation.
        All but the observed values are NaN if there are fewer than three ratios
        with :math:`x \geq 2`.

    Note
    ----
    The ratios :math:`r_x = (x + 1) f_{x+1} / f_x` do not depend on the unobserved
    species. They are constant under a Poisson distribution, and increase under
    heterogeneity. Their logarithms are linear in :math:`\log(x + 1)` for the
    Poisson, the geometric and the Conway-Maxwell-Poisson distribution. This line is
    fitted to the ratios with :math:`x \geq 2`, by least squares weighted with the
    inverse variances :math:`1 / f_x + 1 / f_{x+1}`, and extrapolated to :math:`x =
    1`. This is model M1 in Lerdsuwansri et al. (2025, equation 11), in which "excess"
    is the coefficient of an indicator for :math:`x = 1` (without the factor :math:`x +
    1` in the ratios, which only changes the slope). For incidence data the ratios are
    those of the binomial distribution, :math:`T (x + 1) / (T - x) \cdot f_{x+1} / f_x`
    (cf. Böhning 2016).

    Lerdsuwansri et al. compare M1 with M0, the model without the indicator, by AIC or
    BIC, as given here. The test is ours: the Wald test of the indicator, with the
    variances of the ratios as known. In our simulations it keeps its level under the
    Poisson and the geometric distribution, and is conservative under the negative
    binomial, for which the trend is not a line. AIC finds one-inflation more easily: it
    chose M1 for 22 to 45% of their samples without one-inflation (Table 4).

    One-inflation and heterogeneity can look the same: what is tested is whether
    :math:`r_1` is lower than the trend of the other ratios. Tests that assume a
    Poisson or binomial distribution without further heterogeneity reject more often,
    since heterogeneity also gives more singletons than these distributions: they find
    one-inflation in the immigrants and prinia data of our unit tests, the present test
    does not. It also has little power with few ratios.

    References
    ----------
    - D. Böhning, 'Ratio plot and ratio regression with applications to social and
      medical sciences', Statistical Science (2016), 205-218.
    - R. Lerdsuwansri et al., 'Identifying one-inflation in regression models for
      ratio estimators in single-source capture-recapture problems', Journal of
      Statistical Computation and Simulation (2025), 3279-3299.
    """
    f = np.bincount(ds.counts[ds.counts > 0], minlength=3)
    x = np.arange(1, min(len(f) - 1, max_count))
    x = x[(f[x] > 0) & (f[x + 1] > 0)]
    r = (x + 1) * f[x + 1] / f[x]
    if isinstance(ds, IncidenceData):
        r = r * ds.T / (ds.T - x)
    var = 1 / f[x] + 1 / f[x + 1]

    test = dict.fromkeys(["f1", "f1_expected", "r1", "r1_expected", "excess", "z", "p",
                          "aic0", "aic1", "bic0", "bic1"], np.nan)
    test.update(f1=ds.f1, r1=r[0] if 1 in x else np.nan)
    if 1 not in x or len(x) < 4:
        return test

    X = np.column_stack([np.ones(len(x)), np.log(x + 1), x == 1])
    coefs, cov, test["aic1"], test["bic1"] = _regression(X, np.log(r), var)
    _, _, test["aic0"], test["bic0"] = _regression(X[:, :2], np.log(r), var)
    z = coefs[2] / np.sqrt(cov[2, 2])
    r1 = np.exp(coefs[0] + coefs[1] * np.log(2))
    test.update(f1_expected=f[1] * test["r1"] / r1, r1_expected=r1, excess=coefs[2], z=z,
                p=scipy.stats.norm.cdf(z))
    return test


def _regression(X, y, var):
    """Regression weighted with inverse variances: coefficients, their covariances for
    these variances, AIC and BIC (Lerdsuwansri et al. 2025, appendix)."""
    cov = np.linalg.inv(X.T @ (X / var[:, None]))
    coefs = cov @ X.T @ (y / var)
    n, p = X.shape
    sigma2 = np.sum((y - X @ coefs) ** 2 / var) / n
    deviance = n * np.log(2 * np.pi) + np.sum(np.log(sigma2 * var)) + n
    return coefs, cov, deviance + 2 * (p + 1), deviance + np.log(n) * (p + 1)


__all__ = ['one_inflation']
