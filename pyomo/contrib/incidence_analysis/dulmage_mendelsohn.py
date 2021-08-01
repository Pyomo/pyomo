#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.common.dependencies import networkx as nx
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
        dulmage_mendelsohn as dm_nx,
        )

def dulmage_mendelsohn(matrix_or_graph, top_nodes=None, matching=None):
    """
    """
    # TODO: What if matrix/graph is disconnected?
    if isinstance(matrix_or_graph, nx.Graph):
        graph = matrix_or_graph
        partition = dm_nx(graph, top_nodes=top_nodes, matching=matching)
    else:
        # Assume matrix_or_graph is a scipy coo_matrix
        matrix = matrix_or_graph
        M, N = matrix.shape
        nxb = nx.algorithms.bipartite
        from_biadjacency_matrix = nxb.matrix.from_biadjacency_matrix

        # Matrix rows have bipartite=0, columns have bipartite=1
        bg = from_biadjacency_matrix(matrix)
        row_partition, col_partition = dm_nx(bg, top_nodes=list(range(M)))

        partition = (
                row_partition,
                tuple([n-M for n in subset] for subset in col_partition)
                # Column nodes have values in [M, M+N-1]. Apply the offset
                # to get values corresponding to indices in user's matrix.
                )

    return partition
