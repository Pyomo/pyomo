#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import networkx as nx


def get_independent_submatrices(matrix):
    """Partition a matrix into irreducible block diagonal form

    This is equivalent to identifying the connected components of the bipartite
    incidence graph of rows and columns.

    Parameters
    ----------
    matrix: ``scipy.sparse.coo_matrix``
        Matrix to partition into block diagonal form

    Returns
    -------
    row_blocks: list of lists
        Partition of row coordinates into diagonal blocks
    col_blocks: list of lists
        Partition of column coordinates into diagonal blocks

    """
    nxc = nx.algorithms.components
    nxb = nx.algorithms.bipartite
    from_biadjacency_matrix = nxb.matrix.from_biadjacency_matrix
    graph = from_biadjacency_matrix(matrix)
    N, M = matrix.shape
    connected_components = list(nxc.connected_components(graph))
    # connected_components is a list of sets of nodes

    # By convention, row nodes have values in [0, N-1], while column
    # nodes have values in [N, N+M-1].
    # We could also check the "bipartite" attribute of each node...
    row_blocks = [
        sorted([node for node in comp if node < N]) for comp in connected_components
    ]
    col_blocks = [
        sorted([node - N for node in comp if node >= N])
        for comp in connected_components
    ]
    return row_blocks, col_blocks
