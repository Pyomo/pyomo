#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import numpy as np
from scipy.sparse import coo_matrix

def build_bounds_mask(vector):
    """
    Creates masks for converting from the full vector of bounds that 
    may contain -np.inf or np.inf to a vector of bounds that are finite
    only.
    """
    return build_compression_mask_for_finite_values(vector)

def build_compression_matrix(compression_mask):
    """
    Return a sparse matrix CM of ones such that
    compressed_vector = CM*full_vector based on the 
    compression mask
    """
    cols = compression_mask.nonzero()[0]
    nnz = len(cols)
    rows = np.arange(nnz, dtype=np.int)
    data = np.ones(nnz)
    return coo_matrix((data, (rows, cols)), shape=(nnz, len(compression_mask)))
    
def build_compression_mask_for_finite_values(vector):
    """
    Creates masks for converting from the full vector of
    values to the vector that contains only the finite values. This is 
    typically used to convert a vector of bounds (that may contain np.inf
    and -np.inf) to only the bounds that are finite.
    """
    full_finite_mask = np.isfinite(vector)
    return full_finite_mask

# TODO: Is this needed anywhere?
#def build_expansion_map_for_finite_values(vector):
#    """
#    Creates a map from the compressed vector to the full
#    vector based on the locations of finite values only. This is 
#    typically used to map a vector of bounds (that is compressed to only
#    contain the finite values) to a full vector (that may contain np.inf
#    and -np.inf).
#    """
#    full_finite_mask = np.isfinite(vector)
#    finite_full_map = full_finite_mask.nonzero()[0]
#    return finite_full_map
    
def full_to_compressed(full_array, compression_mask, out=None):
    if out is not None:
        np.compress(compression_mask, full_array, out=out)
        return out
    else:
        return np.compress(compression_mask, full_array)

def compressed_to_full(compressed_array, compression_mask, out=None, default=None):
    if out is None:
        ret = np.empty(len(compression_mask))
        ret.fill(np.nan)
    else:
        ret = out

    ret[compression_mask] = compressed_array
    if default is not None:
        ret[~compression_mask] = default

    return ret


def extract_submatrix(M, row_indices, col_indices):
    """
    Return the submatrix that corresponds to the list
    of row and column indices specified

    Parameters
    ----------

    row_indices : np.array with indices of rows to extract
    col_indices : np.array with indices of rows to extract
    """
    matrix = M.tocoo()
    row_mask = np.isin(matrix.row, row_indices)
    col_mask = np.isin(matrix.col, col_indices)
    submatrix_mask = row_mask & col_mask
    submatrix_irows = np.compress(submatrix_mask, matrix.row)
    submatrix_jcols = np.compress(submatrix_mask, matrix.col)
    submatrix_data = np.compress(submatrix_mask, matrix.data)

    # ToDo: this is expensive - have to think about how to do this with numpy
    row_submatrix_map = {j: i for i, j in enumerate(row_indices)}
    for i, v in enumerate(submatrix_irows):
        submatrix_irows[i] = row_submatrix_map[v]

    col_submatrix_map = {j: i for i, j in enumerate(col_indices)}
    for i, v in enumerate(submatrix_jcols):
        submatrix_jcols[i] = col_submatrix_map[v]

    return coo_matrix((submatrix_data, (submatrix_irows, submatrix_jcols)),
                      shape=(len(row_indices), len(col_indices)))