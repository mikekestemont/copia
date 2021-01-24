from collections import Counter

import numpy as np

def to_abundance(species):
    """
    """
    return np.array(tuple(Counter(species).values()),
           dtype=np.int)


