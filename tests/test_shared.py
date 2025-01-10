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

    without_ci = copia.estimators.shared_richness(s1, s2, CI=False)
    print(without_ci)

    assert int(without_ci['total']) == 389
    assert int(without_ci['unobs_shared']) == 139
    assert int(without_ci['f0+']) == 48
    assert int(without_ci['f+0']) == 68
    assert int(without_ci['f00']) == 22