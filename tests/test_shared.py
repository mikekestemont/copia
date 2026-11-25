import pytest
import warnings

import numpy as np
from copia.data import to_copia_dataset
import copia.estimators
import copia.diversity
import copia.stats
import copia.rarefaction_extrapolation
import copia.utils as u

def test_shared():
    """
    The data used for testing was discussed (cd. Table 1) in:    
    > Chao, Anne, Chiu Chun‐Huo, Robert K. Colwell, Luiz Fernando S. Magnago, Robin L. Chazdon,
    > and Nicholas J. Gotelli. 2017. Deciphering the Enigma of Undetected Species, Phylogenetic,
    > and Functional Diversity Based on Good‐Turing Theory. *Ecology* 98 (11): 2914–29.

    The data (tree species abundance data for the edge and interior habitats in forest fragments
    of south-eastern Brazil) are borrowed from another study:

    > Magnago, L. F. S., D. P. Edwards, F. A. Edwards, A. Magrach, S. V. Martins, and W. F.
    > Laurance. 2014. Functional attributes change but functional richness is unchanged after
    > fragmentation of Brazilian Atlantic forests. *Journal of Ecology* 102:475−485.

    This data was retrieved from Anne Chao's [Github page](https://raw.githubusercontent.com/AnneChao/Good-Turing/refs/heads/master/DataS1%20(abundance%20data).txt).

    """
    import pandas as pd
    trees = pd.read_csv('datasets/trees.csv', header=0, sep=' ').reset_index(drop=True)
    trees.columns = ['s1', 's2']
    trees = trees.reset_index(names=['species'])
    trees['s1'] = trees['s1'].astype(int) # Edge trees
    trees['s2'] = trees['s2'].astype(int) # Interior trees

    s1 = to_copia_dataset(trees, data_type='abundance', input_type='counts',
                          index_column='species', count_column='s1', remove_zeros=False)
    s2 = to_copia_dataset(trees, data_type='abundance', input_type='counts',
                          index_column='species', count_column='s2', remove_zeros=False)
    
    # verify integrity of basic stats:
    assert s1.S_obs == 319
    assert s1.f1 == 110
    assert s1.f2 == 48
    assert s1.n == 1794

    assert s2.S_obs == 356
    assert s2.f1 == 123
    assert s2.f2 == 48
    assert s2.n == 2074

    # make sure the counts are still aligned:
    assert len(trees) == len(s1.counts)
    assert len(trees) == len(s2.counts)
    print(s1)
    print(s2)

    # test point estimates:
    without_ci = copia.estimators.chao_shared(s1, s2, CI=False)

    assert int(without_ci['total']) == 389
    assert int(without_ci['unobs_shared']) == 139
    assert int(without_ci['f0+']) == 48
    assert int(without_ci['f+0']) == 68
    assert int(without_ci['f00']) == 22

    # test the confidence intervals:
    results = copia.estimators.chao_shared(s1, s2, CI=True, conf=.95,
                                               n_iter=100000, seed=573861)
    print(results)
    
    # format the results for inspection:
    data = {
        'name': ['total', 'obs_shared', 'unobs_shared', 'f0+', 'f+0', 'f00'],
        'Est': [
            results['total'],
            results['obs_shared'],
            results['unobs_shared'],
            results['f0+'],
            results['f+0'],
            results['f00'],
        ]
    }
    
    if 'CI' in results:
        data['se'] = [
            results['se']['total'],
            results['se']['obs_shared'],
            results['se']['unobs_shared'],
            results['se']['f0+'],
            results['se']['f+0'],
            results['se']['f00'],
        ]
        data['95% LCL'] = [
            results['CI']['total']['lower'],
            results['CI']['obs_shared']['lower'],
            results['CI']['unobs_shared']['lower'],
            results['CI']['f0+']['lower'],
            results['CI']['f+0']['lower'],
            results['CI']['f00']['lower'],
        ]
        data['95% UCL'] = [
            results['CI']['total']['upper'],
            results['CI']['obs_shared']['upper'],
            results['CI']['unobs_shared']['upper'],
            results['CI']['f0+']['upper'],
            results['CI']['f+0']['upper'],
            results['CI']['f00']['upper'],
        ]
    
    df = pd.DataFrame(data)
    print(df)

    for idx, row in df.iterrows():
        name = row['name']
        estimate = row['Est']
        lcl = row['95% LCL']
        ucl = row['95% UCL']
        
        # check whether point estimates are within CIs:
        assert lcl <= estimate <= ucl, (
            f"CI check failed for {name}: "
            f"estimate ({estimate:.2f}) not in "
            f"[{lcl:.2f}, {ucl:.2f}]"
        )

        # check whether CIs are >=0:
        assert lcl >= 0, (
            f"CI check failed for {name}, LCL: "
            f"estimate ({lcl:.2f}) < 0"
        )
        assert ucl >= 0, (
            f"CI check failed for {name}, UCL: "
            f"estimate ({ucl:.2f})"
        )


def shared_example(data_type='abundance', T=(None, None)):
    s1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 0, 0, 1, 2, 4, 1, 1, 2, 2])
    s2 = np.array([1, 1, 2, 1, 3, 2, 2, 1, 1, 1, 4, 2, 1, 0, 0, 2, 1, 2, 2, 3])
    return [to_copia_dataset(s, data_type=data_type, input_type='counts',
                             n_sampling_units=t, remove_zeros=False)
            for s, t in zip((s1, s2), T)]


def test_shared_seed():
    s1, s2 = shared_example()
    kwargs = dict(CI=True, n_iter=50)
    reference = copia.estimators.chao_shared(s1, s2, seed=1, **kwargs)
    # the seed alone fixes the result, whatever the state of the global generator
    for global_seed in range(5):
        np.random.seed(global_seed)
        assert copia.estimators.chao_shared(s1, s2, seed=1, **kwargs) == reference
    assert copia.estimators.chao_shared(s1, s2, seed=np.int64(1), **kwargs) == reference
    assert copia.estimators.chao_shared(s1, s2, seed=2, **kwargs) != reference

    # the wrapper passes its arguments on
    def wrapper(**kw):
        return copia.estimators.diversity(s1, s2, method='chao_shared', CI=True, **kw)
    assert wrapper(n_iter=50, seed=1) == reference
    assert wrapper(n_iter=60, seed=1) != reference
    assert wrapper(n_iter=50, seed=1, conf=0.5)['CI'] != reference['CI']
