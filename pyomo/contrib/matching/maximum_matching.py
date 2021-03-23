#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import networkx as nx
import networkx.algorithms.bipartite as nxb
import networkx.algorithms.components as nxc
from networkx.algorithms.bipartite.matrix import (
        from_biadjacency_matrix,
        )

def maximum_matching(matrix):
    """
    Returns a maximum matching of the matrix as a dict
    from row indices to column indices.
    """
    M, N = matrix.shape
    bg = from_biadjacency_matrix(matrix)

    # Check assumptions regarding from_biadjacency_matrix function:
    for i in range(M):
        # First M nodes in graph correspond to rows
        assert bg.nodes[i]['bipartite'] == 0

    for j in range(M, M+N):
        # Last N nodes in graph correspond to columns
        assert bg.nodes[j]['bipartite'] == 1

    # If the matrix is block diagonal, the graph will be disconnected.
    # This is fine, but we need to separate into connected components
    # for NetworkX to not complain.
    conn_comp = [bg.subgraph(c) for c in nxc.connected_components(bg)]
    matchings = [nxb.maximum_matching(c) for c in conn_comp]
    # If n0 < M, then n1 >= M. n0 is the row index, n1-M is the column index
    max_matching = {
            n0: n1-M for m in matchings for n0, n1 in m.items() if n0 < M
            }
    return max_matching
