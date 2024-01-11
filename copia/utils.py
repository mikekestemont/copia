"""
Various utility functions.
"""
import multiprocessing as mp

import numpy as np
from tqdm.auto import tqdm


def valid_count_data(x):
    if (x < 0).any():
        msg = "Elements of `x` should be strictly non-negative"
        raise ValueError(msg)

    if x.sum() <= 0:
        msg = "`x` should contain at least 1 sighting"
        raise ValueError(msg)
    
    return True


class Parallel:
    r"""
    Helper class for parallel execution.
    """
    def __init__(self, n_workers, n_tasks, disable_pb=False):
        self.pool = mp.Pool(n_workers)
        self._results = []
        self._pb = tqdm(total=n_tasks, disable=disable_pb)

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


__all__ = ['Parallel', 'check_random_state']
