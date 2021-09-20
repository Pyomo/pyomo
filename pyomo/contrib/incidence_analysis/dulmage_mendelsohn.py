#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
from pyomo.common.dependencies import networkx as nx
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
    dulmage_mendelsohn as dm_nx,
    )
"""
This module imports the general Dulmage-Mendelsohn-on-a-graph function
from "common" and implements an interface for coo_matrix-like objects.
"""

RowPartition = namedtuple(
        "RowPartition",
        ["unmatched", "overconstrained", "underconstrained", "square"],
        )
ColPartition = namedtuple(
        "ColPartition",
        ["unmatched", "underconstrained", "overconstrained", "square"],
        )

def dulmage_mendelsohn(matrix_or_graph, top_nodes=None, matching=None):
    """
    COO matrix or NetworkX graph interface to the coarse Dulmage Mendelsohn
    partition. The matrix or graph should correspond to a Pyomo model.
    top_nodes must be provided if a NetworkX graph is used, and should
    correspond to Pyomo constraints.

    """
    if isinstance(matrix_or_graph, nx.Graph):
        # The purpose of handling graphs here is that if we construct NX graphs
        # directly from Pyomo expressions, we can eliminate the overhead of
        # convering expressions to a matrix, then the matrix to a graph.
        #
        # In this case, top_nodes should correspond to constraints.
        graph = matrix_or_graph
        if top_nodes is None:
            raise ValueError(
                    "top_nodes must be specified if a graph is provided,"
                    "\notherwise the result is ambiguous."
                    )
        partition = dm_nx(graph, top_nodes=top_nodes, matching=matching)
        # RowPartition and ColPartition do not make sense for a general graph.
        # However, here we assume that this graph comes from a Pyomo model,
        # and that "top nodes" are constraints.
        partition = (RowPartition(*partition[0]), ColPartition(*partition[1]))
    else:
        # Assume matrix_or_graph is a scipy coo_matrix
        matrix = matrix_or_graph
        M, N = matrix.shape
        nxb = nx.algorithms.bipartite
        from_biadjacency_matrix = nxb.matrix.from_biadjacency_matrix

        if matching is not None:
            # If a matching was provided for a COO matrix, we assume it
            # maps row indices to column indices, for compatibility with
            # output of our maximum_matching function.

            # NetworkX graph has column nodes offset by M
            matching = {i: j + M for i, j in matching.items()}
            inv_matching = {j: i for i, j in matching.items()}
            # DM function requires matching map to contain inverse matching too
            matching.update(inv_matching)

        # Matrix rows have bipartite=0, columns have bipartite=1
        bg = from_biadjacency_matrix(matrix)
        row_partition, col_partition = dm_nx(
                bg,
                top_nodes=list(range(M)),
                matching=matching,
                )

        partition = (
                row_partition,
                tuple([n - M for n in subset] for subset in col_partition)
                # Column nodes have values in [M, M+N-1]. Apply the offset
                # to get values corresponding to indices in user's matrix.
                )

    partition = (RowPartition(*partition[0]), ColPartition(*partition[1]))
    return partition
