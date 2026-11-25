import functools
import inspect

import numpy as np
import pandas as pd

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class CopiaData:
    S_obs: int
    f1: int
    f2: int
    n: int
    counts: np.ndarray

    @property
    def sample_size(self):
        """The sample size that enters the estimators: the number of individuals
        (n) for abundance data, the number of sampling units (T) for incidence data."""
        return self.n


@dataclass(slots=True, frozen=True)
class AbundanceData(CopiaData):
    pass


@dataclass(slots=True, frozen=True)
class IncidenceData(CopiaData):
    T: int

    @property
    def sample_size(self):
        return self.T


@dataclass(slots=True, frozen=True)
class ListData(IncidenceData):
    """Items on several lists, with the pattern of presence and absence of every
    item. As incidence data, the lists are the sampling units."""
    lists: tuple            # names of the lists
    patterns: np.ndarray    # observed patterns, of shape (n patterns, n lists)
    freqs: np.ndarray       # number of items with each pattern


def accepts(*types):
    """Declare the types of data that an estimator is made for. The decorated
    function refuses other data (for its first argument)."""
    def decorate(fn):
        first = next(iter(inspect.signature(fn).parameters))

        @functools.wraps(fn)
        def checked(*args, **kwargs):
            if args or first in kwargs:
                ds = args[0] if args else kwargs[first]
                if not isinstance(ds, types):
                    needed = " or ".join(t.__name__ for t in types)
                    raise TypeError(f"{fn.__name__} needs {needed}, not {type(ds).__name__}")
            return fn(*args, **kwargs)
        return checked
    return decorate


def to_abundance_counts(data, input_type="observations", index_column=None,
                        count_column=None):
    """
    Converts abundance data into a format suitable for analysis, accommodating
    various input structures.

    This function is tailored for processing abundance data, which indicates the
    presence and frequency of items in a dataset. It supports two primary input
    types: raw observations and direct count data.

    Parameters:
    data : list, tuple, np.ndarray, pd.Series, pd.DataFrame, dict
        The abundance data to be processed. The format of this data depends on
        the input_type.
    input_type : str, optional
        Specifies the format of the input data. Two options are available:
        - 'observations': For raw observational data provided as a list, tuple,
          NumPy array, pandas Series, or DataFrame.
        - 'counts': For data where counts are directly provided.
    index_column : str, optional
        The name of the column in a pandas DataFrame input that represents items.
        Required when input_type is 'observations' and data is a DataFrame.
    count_column : str, optional
        The name of the column in a pandas DataFrame input that represents counts.
        Required when input_type is 'counts' and data is a DataFrame.

    Returns:
    A pandas Series
        The series contains the unique items (index) and their corresponding counts
        (values) based on the specified input_type.

    Raises:
    ValueError:
        If input_type is not one of the specified options or if the required columns
        for DataFrame input are not specified.
    TypeError:
        If the data format does not match the specified input_type.

    Example Usage:
    >>> data_list = ['apple', 'banana', 'apple', 'orange']
    >>> to_abundance_counts(data_list, input_type='observations')
    # Processes the list and returns counts of each unique fruit

    >>> data_frame = pd.DataFrame({'fruit': ['apple', 'banana'], 'count': [2, 3]})
    >>> to_abundance_counts(
    ...     data_frame, input_type='counts', index_column='fruit',
    ...     count_column='count')
    # Directly uses the provided counts from the DataFrame

    >>> data_dict = {'apple': 4, 'banana': 5}
    >>> to_abundance_counts(data_dict, input_type='counts')
    # Directly uses the provided counts from the dictionary
    """
    if input_type not in ("observations", "counts"):
        raise ValueError("input_type must be either 'observations' or 'counts'")

    if input_type == "observations":
        match data:
            case list() | tuple() | np.ndarray() | pd.Series():
                counts = pd.Series(data).value_counts()
            case pd.DataFrame():
                if index_column is None:
                    raise ValueError(
                        "index_column must be specified for DataFrame input")
                counts = data[index_column].value_counts()
            case _:
                raise TypeError(
                    "For 'observations', data should be a list, tuple, ndarray, Series, or DataFrame.")

    else:
        match data:
            case list() | tuple() | np.ndarray() | pd.Series():
                counts = pd.Series(data)
            case pd.DataFrame():
                if index_column is None or count_column is None:
                    raise ValueError(
                        "Both index_column and count_column must be specified for DataFrame input")
                counts = data.set_index(index_column)[count_column]
            case dict():
                counts = pd.Series(data)
            case _:
                raise TypeError(
                    "For 'counts', data should be a list, tuple, ndarray, Series, DataFrame, or dict.")

    return counts


def to_incidence_counts(
    data,
    input_type="observation_matrix",
    index_column=None,
    location_column=None,
    count_column=None):
    """
    Converts incidence data into a format suitable for analysis, handling various
    types of input structures.

    This function is designed to process incidence data, which represents the
    occurrences of items across different locations. It supports three primary
    input types: observation matrices, observation lists, and direct count data.

    Parameters:
    data : np.ndarray, pd.DataFrame, list, tuple, or dict
        The incidence data to be processed. The format of this data depends on
        the input_type.
    input_type : str, optional
        Specifies the format of the input data. Three options are available:
        - 'observation_matrix': For data provided as a matrix (NumPy array or
          pandas DataFrame) where rows (or columns) represent unique items and
          columns (or rows) represent unique locations. Non-zero entries indicate
          the presence of an item in a location.
        - 'observation_list': For data provided as a list, tuple, or NumPy array
          of (item, location) pairs, or a pandas DataFrame with specified columns
          for items and locations.
        - 'counts': For data where counts are directly provided.
    index_column : str, optional
        The name of the column in a pandas DataFrame input that represents items.
        Required when input_type is 'observation_list' or 'counts' and data is a
        DataFrame.
    location_column : str, optional
        The name of the column in a pandas DataFrame input that represents locations.
        Required when input_type is 'observation_list'.
    count_column : str, optional
        The name of the column in a pandas DataFrame input that represents counts.
        Required when input_type is 'counts' and data is a DataFrame.

    Returns:
    A pandas Series
        The series contains the counts of unique items (index) and their corresponding
        counts (values) based on the specified input_type.

    Raises:
    ValueError:
        If input_type is not one of the specified options.
    TypeError:
        If the data format does not match the specified input_type.

    Example Usage:
    >>> data_matrix = np.array([[1, 0], [0, 1], [1, 1]])
    >>> to_incidence_counts(data_matrix, input_type='observation_matrix')
    # Returns counts of non-zero entries in each row of the matrix

    >>> data_list = [('item1', 'loc1'), ('item2', 'loc2'), ('item1', 'loc2')]
    >>> to_incidence_counts(data_list, input_type='observation_list')
    # Processes list of item-location pairs and returns counts of unique locations for each item

    >>> data_counts = {'item1': 5, 'item2': 3}
    >>> to_incidence_counts(data_counts, input_type='counts')
    # Directly uses the provided counts
    """
    if input_type not in ("observation_matrix", "observation_list", "counts"):
        raise ValueError(
            "input_type must be either 'observation_matrix', 'observation_list', or 'counts'"
        )

    if input_type == "observation_matrix":
        match data:
            case np.ndarray():
                counts = pd.DataFrame(data).apply(np.count_nonzero, axis=1)
            case pd.DataFrame():
                counts = data.apply(np.count_nonzero, axis=1)
            case _:
                raise TypeError(
                    "For 'matrix', data should be a numpy ndarray or a pandas DataFrame."
                )

    elif input_type == "observation_list":
        match data:
            case list() | tuple() | np.ndarray() if len(data[0]) == 2:
                data = pd.DataFrame(data, columns=["item", "location"])
                counts = data.groupby("item")["location"].nunique()
            case pd.DataFrame():
                if index_column is None or location_column is None:
                    raise ValueError(
                        "index_column and location_column must be specified for DataFrame input")
                counts = data.groupby(index_column)[location_column].nunique()
            case _:
                raise TypeError("For 'dataframe', data must be a pandas DataFrame.")

    else:
        match data:
            case list() | tuple() | np.ndarray() | pd.Series():
                counts = pd.Series(data)
            case pd.DataFrame():
                if index_column is None or count_column is None:
                    raise ValueError(
                        "Both index_column and count_column must be specified for DataFrame input")
                counts = data.set_index(index_column)[count_column]
            case dict():
                counts = pd.Series(data)
            case _:
                raise TypeError(
                    "For 'counts', data should be a list, tuple, ndarray, Series, DataFrame, or dict.")

    return counts


def to_list_patterns(
    data,
    input_type="observation_matrix",
    index_column=None,
    location_column=None,
    count_column=None):
    """
    Converts data on several lists into the numbers of items with each pattern of
    presence (1) and absence (0) on the lists.

    Parameters:
    data : np.ndarray, pd.DataFrame, pd.Series, list, tuple, or dict
        The data on the lists, in a format that depends on the input_type.
    input_type : str, optional
        Specifies the format of the input data. Three options are available:
        - 'observation_matrix': A matrix (NumPy array or pandas DataFrame) with one
          row per item and one column per list. Non-zero entries indicate the
          presence of an item on a list.
        - 'observation_list': A list, tuple, or NumPy array of (item, list) pairs,
          or a pandas DataFrame with specified columns for items and lists.
        - 'counts': The number of items with each pattern, as published for most
          studies: a dict or pandas Series with the patterns as keys ('101', or
          (1, 0, 1)), or a DataFrame with one column per list and a count column.
    index_column : str, optional
        The name of the column that represents items. Required when input_type is
        'observation_list' and data is a DataFrame.
    location_column : str, optional
        The name of the column that represents lists. Required when input_type is
        'observation_list' and data is a DataFrame.
    count_column : str, optional
        The name of the column that represents counts. Required when input_type is
        'counts' and data is a DataFrame.

    Returns:
    A pandas DataFrame with one column per list, and a pandas Series with the number
    of items with each pattern. Items that are on none of the lists are left out.

    Example Usage:
    >>> to_list_patterns({'11': 5, '10': 8, '01': 3}, input_type='counts')
    """
    if input_type not in ("observation_matrix", "observation_list", "counts"):
        raise ValueError(
            "input_type must be either 'observation_matrix', 'observation_list', or 'counts'"
        )

    if input_type == "observation_matrix":
        matrix = pd.DataFrame(data)
        freqs = pd.Series(1, index=matrix.index)

    elif input_type == "observation_list":
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=["item", "list"])
            index_column, location_column = "item", "list"
        elif index_column is None or location_column is None:
            raise ValueError(
                "index_column and location_column must be specified for DataFrame input")
        matrix = pd.crosstab(data[index_column], data[location_column])
        freqs = pd.Series(1, index=matrix.index)

    elif isinstance(data, pd.DataFrame):
        if count_column is None:
            raise ValueError("count_column must be specified for DataFrame input")
        matrix, freqs = data.drop(columns=count_column), data[count_column]

    else:
        freqs = pd.Series(data)
        matrix = pd.DataFrame([[int(p) for p in pattern] for pattern in freqs.index],
                              index=freqs.index)

    if matrix.shape[1] < 2:
        raise ValueError("at least two lists are needed")
    if (freqs < 0).any():
        raise ValueError("the numbers of items should be non-negative")
    if matrix.columns.equals(pd.RangeIndex(matrix.shape[1])):
        matrix.columns = range(1, matrix.shape[1] + 1)
    matrix = (matrix.fillna(0) != 0).astype(int)
    keep = (matrix.sum(axis=1) > 0) & (freqs > 0)
    freqs = freqs[keep].groupby([matrix.loc[keep, c] for c in matrix.columns]).sum()
    return freqs.index.to_frame(index=False), freqs.reset_index(drop=True)


def to_copia_dataset(
    observations,
    data_type="abundance",
    input_type="observations",
    index_column=None,
    count_column=None,
    location_column=None,
    n_sampling_units=None,
    remove_zeros=True):
    """
    Converts the given observations into a Copia dataset, accommodating
    abundance data, incidence data, and data on several lists.

    Parameters:
    observations : pd.DataFrame
        The input data to be converted into a Copia dataset.
    data_type : str, optional
        The type of data in 'observations'. Can be 'abundance', 'incidence' or
        'lists'.
    input_type : str, optional
        The format of the input data (e.g., 'observations', 'counts'). Required
        for both abundance and incidence data.
    index_column : str, optional
        The name of the column in 'observations' representing the items.
        Required for certain input types.
    count_column : str, optional
        The name of the column representing counts in 'observations'.
        Required for 'counts' input type.
    location_column : str, optional
        The name of the column in 'observations' representing locations (or lists).
        Required for 'observation_list' input type in incidence data and lists.
    title : str, optional
        The title for the dataset.
    description : str, optional
        A description for the dataset.
    n_sampling_units : int, optional
        The number of sampling units (T). Required for incidence data.
    remove_zeros : bool, optional (default = True)
        Whether to remove zero counts from ds.counts (set to False for aligned datasets)
    
    Returns:
    An copia.CopiaData
        The dataset contains counts, and various statistics relevant for Copia analysis.

    Example Usage:
    >>> df = pd.DataFrame(
    ...     {
    ...         'species': ['sparrow', 'eagle', 'shark', 'beetle'],
    ...         'count': [15, 3, 1, 2]
    ...     })
    >>> ds = to_copia_dataset(
    ...     df, data_type='abundance', input_type='counts',
    ...     index_column='species', count_column='count')
    # Creates a Copia dataset from abundance data with counts.
    """    
    if data_type.lower() == "lists":
        patterns, freqs = to_list_patterns(
            observations, input_type, index_column, location_column, count_column)
        on = np.repeat(patterns.sum(axis=1).values, freqs)
        return ListData(
            S_obs=len(on), f1=np.count_nonzero(on == 1), f2=np.count_nonzero(on == 2),
            n=on.sum(), counts=on, T=patterns.shape[1], lists=tuple(patterns.columns),
            patterns=patterns.values, freqs=freqs.values)

    match data_type.lower():
        case "abundance":
            counts = to_abundance_counts(
                observations, input_type, index_column, count_column)
        case "incidence":
            counts = to_incidence_counts(
                observations, input_type, index_column, location_column, count_column)
        case _:
            raise TypeError(
                f"Data type {data_type} is invalid")
            
    n = counts.sum()
    
    counts = counts.values
    # Compute some basic statistics that are used in many of copia's functions
    if remove_zeros:
        counts = counts[counts > 0]

    f1 = np.count_nonzero(counts == 1)
    f2 = np.count_nonzero(counts == 2)
    S_obs = counts[counts > 0].shape[0]

    if data_type.strip().lower() == 'incidence':
        if n_sampling_units is None:
            raise ValueError("n_sampling_units is required for incidence data")
        ds = IncidenceData(
            S_obs=S_obs, f1=f1, f2=f2, T=n_sampling_units, n=n, counts=counts)
    else:
        ds = AbundanceData(S_obs=S_obs, f1=f1, f2=f2, n=n, counts=counts)
    
    return ds
