import heapq
import math
import random

import numpy as np


def fisher_logseries():
    pass


class WrightFisher:
    def __init__(self, n_agents, timesteps, warmup=1000, random_state=None):
        self.n_agents = n_agents
        self.timesteps = timesteps
        self.warmup = warmup
        self.rng = np.random.RandomState(random_state)

    def __call__(self, beta, mu):
        n_traits, population = self.n_agents, np.arange(self.n_agents)
        sample = np.zeros((self.timesteps, self.n_agents), dtype=np.int64)

        for timestep in range(self.warmup):
            population, n_traits = self._sample(beta, mu, n_traits, population)

        mu = 0.0 # TODO: Might want to make this a parameter
        for timestep in range(self.timesteps):
            population, n_traits = self._sample(beta, mu, n_traits, population)
            sample[timestep] = population

        return sample

    def _sample(self, beta, mu, n_traits, population):
        traits, counts = np.unique(population, return_counts=True)
        counts = counts ** (1 - beta)
        population = self.rng.choice(
            traits, self.n_agents, replace=True, p=counts / counts.sum()
        )
        innovators = self.rng.rand(self.n_agents) < mu
        n_innovations = innovators.sum()
        population[innovators] = np.arange(n_traits, n_traits + n_innovations)
        return population, n_traits + n_innovations


def simulate_collection(counts, k, beta):
    p = counts ** (1 - beta)
    items = [i for i, c in enumerate(counts) for _ in range(c)]
    random.shuffle(items)
    return np.bincount(heapq.nsmallest(
        k, items,  key=lambda item: -math.log(random.random()) / (p[item] / counts[item])))
