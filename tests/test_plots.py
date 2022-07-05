import pytest
import warnings

import numpy as np
import matplotlib.pyplot as plt

from copia.plot import *
from copia.estimators import diversity
from copia.stats import survival_ratio, species_accumulation
from copia.diversity import hill_numbers, evenness


def test_single_assemblage():
    # example data:
    # https://cran.r-project.org/web/packages/iNEXT/vignettes/Introduction.html
    spider_girdled = [46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    abundance = np.array(spider_girdled)

    ax = abundance_barplot(abundance)
    assert isinstance(ax, plt.Axes)
    
    ax = abundance_histogram(abundance)
    assert isinstance(ax, plt.Axes)

    diagn = diversity(abundance, method='minsample', 
                        solver='grid', CI=False,
                        diagnostics=True)    
    ax = minsample_diagnostic_plot(abundance, diagn)
    assert isinstance(ax, plt.Axes)

    accumulation = species_accumulation(abundance, max_steps=4000,
                                        n_iter=10)
    ax = accumulation_curve(abundance, accumulation,
                   xlabel='documents', ylabel='works',
                   title='species accumulation curve')

    assert isinstance(ax, plt.Axes)

    minsample_est = diversity(abundance, method='minsample', 
                          solver='fsolve', CI=True)
    ax = accumulation_curve(abundance, accumulation, title='Minsample included',
                   xlabel='documents', ylabel='works',
                   minsample=minsample_est, xlim=(0, 4000))
    assert isinstance(ax, plt.Axes)

    estimate = diversity(abundance, method='iChao1', CI=True)
    ax = density_plot(estimate)
    assert isinstance(ax, plt.Axes)

    empirical = diversity(abundance, method='empirical')
    ax = density_plot(estimate, empirical)
    assert isinstance(ax, plt.Axes)
    
    survival = survival_ratio(abundance, method='chao1')
    ax = density_plot(survival, xlim=(0, 1))
    assert isinstance(ax, plt.Axes)

    emp, est = hill_numbers(abundance, n_iter=10)
    ax = hill_plot(emp, est)
    assert isinstance(ax, plt.Axes)
    

def test_multiple_assemblages():
    # test full package on spider data (both girdled and logged):
    # https://cran.r-project.org/web/packages/iNEXT/vignettes/Introduction.html
    assemblages = {}
    spider_girdled = [46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2,
                      2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assemblages['girdled'] = np.array(spider_girdled)

    spider_logged = [88, 22, 16, 15, 13, 10, 8, 8, 7, 7, 7, 5,
                     4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assemblages['logged'] = np.array(spider_logged)

    survival = {}
    for category, assemblage in assemblages.items():
        survival[category] = survival_ratio(assemblage, method='chao1')
    
    ax = multi_kde_plot(survival)
    assert isinstance(ax, plt.Axes)

    ax = survival_errorbar(survival)
    assert isinstance(ax, plt.Axes)

    hill_est = {}
    for lang, assemblage in assemblages.items():
        _, est = hill_numbers(assemblage, n_iter=10)
    hill_est[lang] = est
    evennesses = {l:evenness(hill_est[l]) for l in hill_est}
    ax = evenness_plot(evennesses)
    assert isinstance(ax, plt.Axes)
