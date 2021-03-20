import networkx as nx
import networkx.algorithms.bipartite as nxb
import networkx.algorithms.components as nxc
from networkx.algorithms.bipartite.matrix import (
        from_biadjacency_matrix,
        )

from pyomo.contrib.matching.maximum_matching import maximum_matching

def block_triangularize(matrix, matching=None):
    """
    Arguments
    ---------
    matrix: A SciPy sparse matrix
    matching: A perfect matching of rows and columsn, in the form of a dict
              mapping row indices to column indices
    """

    M, N = matrix.shape
    if M != N:
        raise ValueError("block_triangularize does not currently "
           "support non-square matrices. Got matrix with shape %s."
           % matrix.shape
           )
    bg = from_biadjacency_matrix(matrix)

    if matching is None:
        matching = maximum_matching(matrix)
