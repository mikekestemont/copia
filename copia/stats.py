import numpy as np
import scipy.stats


def dbinom(x, size, prob):
    d = scipy.stats.binom(size, prob).pmf(x)
    return 1 if np.isnan(d) else d


def bt_prob(x):
    x, n = x[x > 0], x.sum()
    f1, f2 = (x == 1).sum(), (x == 2).sum()
    C = 1 - f1 / n * (((n - 1) * f1 / ((n - 1) * f1 + 2 * f2)) if f2 > 0 else
                      ((n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)) if f1 > 0 else
                      0)
    W = (1 - C) / np.sum(x / n * (1 - x / n) ** n)
    p = x / n * (1 - W * (1 - x / n) ** n)
    f0 = np.ceil(((n - 1) / n * f1 ** 2 / (2 * f2)) if f2 > 0 else
                 ((n - 1) / n * f1 * (f1 - 1) / 2))
    p0 = (1 - C) / f0
    p = np.hstack((p, np.array([p0 for i in np.arange(f0)])))
    return p


def bootstrap(x, fn, n_iter=1000, conf=.95):
    pro = fn(x)
    
    p, n = bt_prob(x), x.sum()
    data_bt = np.random.multinomial(n, p, n_iter)
    
    bt_pro = np.array([fn(row) for row in data_bt])
    pro_mean = bt_pro.mean(0)
    
    lci_pro = -np.quantile(bt_pro, (1 - conf) / 2, axis=0) + pro_mean
    uci_pro = np.quantile(bt_pro, 1 - (1 - conf) / 2, axis=0) - pro_mean
    sd_pro = np.std(bt_pro, axis=0)

    bt_pro = pro_mean - bt_pro

    lci_pro, uci_pro = pro - lci_pro, pro + uci_pro
    bt_pro = pro - bt_pro

    return {'richness': pro,
            'lci': lci_pro,
            'uci': uci_pro,
            'std': sd_pro,
            'bootstrap': bt_pro}


def quantile(x, q, weights=None):
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()