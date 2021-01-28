from collections import Counter
import multiprocessing as mp

import numpy as np
import scipy.stats as stats
from scipy.special import gammaln
import tqdm

def to_abundance(species):
    return np.array(tuple(Counter(species).values()),
           dtype=np.int)


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


def dbinom(x, size, prob):
    d = stats.binom(size, prob).pmf(x)
    return 1 if np.isnan(d) else d


def lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)



