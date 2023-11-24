#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
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
    "RowPartition", ["unmatched", "overconstrained", "underconstrained", "square"]
)
"""Named tuple containing the subsets of the Dulmage-Mendelsohn partition
when applied to matrix rows (constraints).

"""

ColPartition = namedtuple(
    "ColPartition", ["unmatched", "underconstrained", "overconstrained", "square"]
)
"""Named tuple containing the subsets of the Dulmage-Mendelsohn partition
when applied to matrix columns (variables).

"""


def dulmage_mendelsohn(matrix_or_graph, top_nodes=None, matching=None):
    """Partition a bipartite graph or incidence matrix according to the
    Dulmage-Mendelsohn characterization

    The Dulmage-Mendelsohn partition tells which nodes of the two bipartite
    sets *can possibly be* unmatched after a maximum cardinality matching.
    Applied to an incidence matrix, it can be interpreted as partitioning
    rows and columns into under-constrained, over-constrained, and
    well-constrained subsystems.

    As it is often useful to explicitly check the unmatched rows and columns,
    ``dulmage_mendelsohn`` partitions rows into the subsets:

    - **underconstrained** - The rows matched with *possibly* unmatched
      columns (unmatched and underconstrained columns)
    - **square** - The well-constrained rows, which are matched with
      well-constrained columns
    - **overconstrained** - The matched rows that *can possibly be* unmatched
      in some maximum cardinality matching
    - **unmatched** - The unmatched rows in a particular maximum cardinality
      matching

    and partitions columns into the subsets:

    - **unmatched** - The unmatched columns in a particular maximum cardinality
      matching
    - **underconstrained** - The columns that *can possibly be* unmatched in
      some maximum cardinality matching
    - **square** - The well-constrained columns, which are matched with
      well-constrained rows
    - **overconstrained** - The columns matched with *possibly* unmatched
      rows (unmatched and overconstrained rows)

    While the Dulmage-Mendelsohn decomposition does not specify an order within
    any of these subsets, the order returned by this function preserves the
    maximum matching that is used to compute the decomposition. That is, zipping
    "corresponding" row and column subsets yields pairs in this maximum matching.
    For example:

    .. doctest::
       :hide:
       :skipif: not (networkx_available and scipy_available)

       >>> # Hidden code block to make the following example runnable
       >>> import scipy.sparse as sps
       >>> from pyomo.contrib.incidence_analysis.dulmage_mendelsohn import dulmage_mendelsohn
       >>> matrix = sps.identity(3)

    .. doctest::
       :skipif: not (networkx_available and scipy_available)

       >>> row_dmpartition, col_dmpartition = dulmage_mendelsohn(matrix)
       >>> rdmp = row_dmpartition
       >>> cdmp = col_dmpartition
       >>> matching = list(zip(
       ...     rdmp.underconstrained + rdmp.square + rdmp.overconstrained,
       ...     cdmp.underconstrained + cdmp.square + cdmp.overconstrained,
       ... ))
       >>> # matching is a valid maximum matching of rows and columns of the matrix!

    Parameters
    ----------
    matrix_or_graph: ``scipy.sparse.coo_matrix`` or ``networkx.Graph``
        The incidence matrix or bipartite graph to be partitioned
    top_nodes: list
        List of nodes in one bipartite set of the graph. Must be provided
        if a graph is provided.
    matching: dict
        A maximum cardinality matching in the form of a dict mapping
        from "top nodes" to their matched nodes *and* from the matched
        nodes back to the "top nodes".

    Returns
    -------
    row_dmp: RowPartition
        The Dulmage-Mendelsohn partition of rows
    col_dmp: ColPartition
        The Dulmage-Mendelsohn partition of columns

    """
    if isinstance(matrix_or_graph, nx.Graph):
        # The purpose of handling graphs here is that if we construct NX graphs
        # directly from Pyomo expressions, we can eliminate the overhead of
        # converting expressions to a matrix, then the matrix to a graph.
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
            bg, top_nodes=list(range(M)), matching=matching
        )

        partition = (
            row_partition,
            tuple([n - M for n in subset] for subset in col_partition)
            # Column nodes have values in [M, M+N-1]. Apply the offset
            # to get values corresponding to indices in user's matrix.
        )

    partition = (RowPartition(*partition[0]), ColPartition(*partition[1]))
    return partition
