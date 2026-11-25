import numpy as np

from copia.coverage import estimate_coverage, sample_coverage
from copia.data import to_copia_dataset


def test_sample_coverage():
    """
    Coverage of rarefied, observed and extrapolated samples of the spider data, as
    published in the vignette of the R package iNEXT (data: Ellison et al. 2010).
    """
    girdled = np.array([46, 22, 17, 15, 15, 9, 8, 6, 6, 4, 2, 2, 2, 2] + [1] * 12)
    logged = np.array([88, 22, 16, 15, 13, 10, 8, 8, 7, 7, 7, 5, 4, 4, 4, 3, 3, 3,
                       3, 2, 2, 2, 2] + [1] * 14)
    for x, published in (
            (girdled, {1: 0.122, 84: 0.900, 168: 0.929, 248: 0.948, 336: 0.964}),
            (logged, {1: 0.145, 126: 0.908, 252: 0.945, 371: 0.958, 504: 0.969})):
        ds = to_copia_dataset(x, input_type="counts")
        for size, coverage in published.items():
            assert round(sample_coverage(ds, size), 3) == coverage
        assert sample_coverage(ds) == sample_coverage(ds, ds.n)

        # the same values in a range of sizes, up to the observed size by default
        steps = estimate_coverage(ds)
        assert len(steps) == ds.n
        assert np.isclose(steps[-1], sample_coverage(ds))
        assert np.allclose(estimate_coverage(ds, max_steps=10, step_size=3),
                           [sample_coverage(ds, size) for size in (1, 4, 7, 10)])


def test_sample_coverage_incidence():
    """
    Ciliates in 19 samples from Etosha Pan, as published in the vignette of iNEXT: the
    frequency counts Q1 to Q10, the observed coverage, and the coverage of extrapolated
    samples. The three larger counts are not given there; only their sum matters.
    """
    Q = (107, 44, 26, 14, 6, 5, 4, 3, 2, 2)
    counts = np.append(np.repeat(range(1, 11), Q), [11, 13, 13])
    ds = to_copia_dataset(counts, data_type="incidence", input_type="counts",
                          n_sampling_units=19)
    assert (ds.S_obs, ds.n) == (216, 516)
    assert round(sample_coverage(ds), 4) == 0.8017
    for size, coverage in ((20, 0.8103610), (88, 0.9909111), (150, 0.9994305)):
        assert round(sample_coverage(ds, size), 7) == coverage
