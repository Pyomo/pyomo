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
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.dependencies import attempt_import
mpi_block_vector, mpi_block_vector_available = attempt_import('pyomo.contrib.pynumero.sparse.mpi_block_vector')


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

    Parameters
    ----------
    compression_mask: np.ndarray or pyomo.contrib.pynumero.sparse.block_vector.BlockVector

    Returns
    -------
    cm: coo_matrix or BlockMatrix
       The compression matrix
    """
    if isinstance(compression_mask, BlockVector):
        n = compression_mask.nblocks
        res = BlockMatrix(nbrows=n, nbcols=n)
        for ndx, block in enumerate(compression_mask):
            sub_matrix = build_compression_matrix(block)
            res.set_block(ndx, ndx, sub_matrix)
        return res
    elif type(compression_mask) is np.ndarray:
        cols = compression_mask.nonzero()[0]
        nnz = len(cols)
        rows = np.arange(nnz, dtype=np.int)
        data = np.ones(nnz)
        return coo_matrix((data, (rows, cols)), shape=(nnz, len(compression_mask)))
    elif isinstance(compression_mask, mpi_block_vector.MPIBlockVector):
        from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
        n = compression_mask.nblocks
        rank_ownership = np.ones((n, n), dtype=np.int64) * -1
        for i in range(n):
            rank_ownership[i, i] = compression_mask.rank_ownership[i]
        res = MPIBlockMatrix(nbrows=n, nbcols=n, rank_ownership=rank_ownership, mpi_comm=compression_mask.mpi_comm,
                             assert_correct_owners=False)
        for ndx in compression_mask.owned_blocks:
            block = compression_mask.get_block(ndx)
            sub_matrix = build_compression_matrix(block)
            res.set_block(ndx, ndx, sub_matrix)
        return res


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

def make_lower_triangular_full(lower_triangular_matrix):
    '''
    This function takes a symmetric matrix that only has entries in the
    lower triangle and makes is a full matrix by duplicating the entries
    '''
    mask = lower_triangular_matrix.row != lower_triangular_matrix.col

    row = np.concatenate((lower_triangular_matrix.row, lower_triangular_matrix.col[mask]))
    col = np.concatenate((lower_triangular_matrix.col, lower_triangular_matrix.row[mask]))
    data = np.concatenate((lower_triangular_matrix.data, lower_triangular_matrix.data[mask]))

    return coo_matrix((data, (row, col)), shape=lower_triangular_matrix.shape)

class CondensedSparseSummation(object):
    def __init__(self, list_of_matrices):
        """
        This class is used to perform a summation of sparse matrices
        while retaining the correct and consistent nonzero structure.
        Create the class with the list of matrices you want to sum,
        and the condensed_summation method remains valid as long as
        the structure of the individual matrices is consistent
        """
        self._nz_tuples = None
        self._maps = None
        self._build_maps(list_of_matrices)

    def _build_maps(self, list_of_matrices):
        """
        This method creates the maps that are used in condensed_sum.
        These maps remain valid as long as the nonzero structure of
        the individual matrices does not change
        """
        # get the list of all unique nonzeros across the matrices
        nz_tuples = set()
        for m in list_of_matrices:
            nz_tuples.update(zip(m.row,m.col))
        nz_tuples = sorted(nz_tuples)
        self._nz_tuples = nz_tuples
        self._row, self._col = list(zip(*nz_tuples))
        row_col_to_nz_map = {t:i for i,t in enumerate(nz_tuples)}

        self._shape = None
        self._maps = list()
        for m in list_of_matrices:
            nnz = len(m.data)
            map_row = np.zeros(nnz)
            map_col = np.zeros(nnz)
            for i in range(nnz):
                map_col[i] = i
                map_row[i] = row_col_to_nz_map[(m.row[i], m.col[i])]
            mp = coo_matrix( (np.ones(nnz), (map_row, map_col)), shape=(len(row_col_to_nz_map),nnz) )
            self._maps.append(mp)
            if self._shape is None:
                self._shape = m.shape
            else:
                assert self._shape == m.shape

    def sum(self, list_of_matrices):
        data = np.zeros(len(self._row))
        assert len(self._maps) == len(list_of_matrices)
        for i,mp in enumerate(self._maps):
            data += mp.dot(list_of_matrices[i].data)
        ret = coo_matrix((data, (np.copy(self._row), np.copy(self._col))), shape=self._shape)
        return ret
