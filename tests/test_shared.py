import pytest

import pandas as pd

from copia.data import to_copia_dataset
import copia.estimators

def test_shared():
    """
    The data used for testing was discussed (cf. Table 1) in:    
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
    trees = pd.read_csv('datasets/trees.csv', header=0, sep=' ').reset_index(drop=True)
    trees.columns = ['s1', 's2']
    trees = trees.reset_index(names=['species'])

    trees['s1'] = trees['s1'].astype(int) # Edge trees
    trees['s2'] = trees['s2'].astype(int) # Interior trees

    # Make sure to set `remove_zeros=False` to ensure aligned abundances:
    s1 = to_copia_dataset(trees, data_type='abundance', input_type='counts',
                          index_column='species', count_column='s1', remove_zeros=False)
    s2 = to_copia_dataset(trees, data_type='abundance', input_type='counts',
                          index_column='species', count_column='s2', remove_zeros=False)
    
    # Verify integrity of basic stats:
    assert s1.S_obs == 319
    assert s1.f1 == 110
    assert s1.f2 == 48
    assert s1.n == 1794

    assert s2.S_obs == 356
    assert s2.f1 == 123
    assert s2.f2 == 48
    assert s2.n == 2074

    # Make sure the counts are still aligned:
    assert len(trees) == len(s1.counts)
    assert len(trees) == len(s2.counts)

    # Test point estimates:
    without_ci = copia.estimators.chao_shared(s1, s2, CI=False)

    assert int(without_ci['total']) == 389
    assert int(without_ci['unobs_shared']) == 139
    assert int(without_ci['f0+']) == 48
    assert int(without_ci['f+0']) == 68
    assert int(without_ci['f00']) == 22

    # Test the confidence intervals:
    results = copia.estimators.chao_shared(s1, s2, CI=True, conf=.95,
                                               n_iter=1000, seed=573861)
    
    # Format the results for inspection:
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

    for idx, row in df.iterrows():
        name = row['name']
        estimate = row['Est']
        lcl = row['95% LCL']
        ucl = row['95% UCL']
        
        assert lcl < ucl, (
            f"CI check failed for {name}: "
            f"estimate ({estimate:.2f}) not in "
            f"[{lcl:.2f}, {ucl:.2f}]"
        )
    