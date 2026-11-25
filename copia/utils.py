"""
Various utility functions.
"""
import numbers
import warnings
from concurrent.futures import BrokenExecutor, Future, ProcessPoolExecutor

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
    Helper class for parallel execution. With a single worker, the tasks
    run in the current process.
    """
    def __init__(self, n_workers, n_tasks, disable_pb=False):
        self.pool = None if n_workers == 1 else ProcessPoolExecutor(n_workers)
        self._futures = []
        self._pb = tqdm(total=n_tasks, disable=disable_pb)

    def apply_async(self, fn, args=()):
        if self.pool is None:
            future = Future()
            try:
                future.set_result(fn(*args))
            except Exception as error:
                future.set_exception(error)
        else:
            future = self.pool.submit(fn, *args)
        future.add_done_callback(lambda _: self._pb.update())
        self._futures.append(future)

    def join(self):
        if self.pool is not None:
            self.pool.shutdown()

    def result(self):
        r"""
        The results in the order of the tasks. Tasks that failed are left out
        with a warning; the error is raised if all of them failed or if a
        worker process died.
        """
        errors = [f.exception() for f in self._futures if f.exception()]
        self._pb.close()
        if errors:
            fatal = [e for e in errors if isinstance(e, BrokenExecutor)]
            if fatal or len(errors) == len(self._futures):
                raise (fatal or errors)[0]
            warnings.warn(f"{len(errors)} of {len(self._futures)} tasks failed and "
                          f"were left out (first error: {errors[0]!r})", UserWarning)
        return [f.result() for f in self._futures if not f.exception()]


def check_random_state(seed):
    r"""
    Helper class to manage stable random
    number generators.
    """
    if seed is np.random:
        return np.random.mtrand._rand
    if seed is None:
        return np.random.RandomState()
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    if isinstance(seed, np.random.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState" " instance" % seed
    )


__all__ = ['Parallel', 'check_random_state']
