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

    len_matching = len(matching)
    if len_matching != M:
        raise ValueError("block_triangularize only supports matrices "
                "that have a perfect matching of rows and columns. "
                "Cardinality of maximal matching is %s" % len_matching
                )

    # Construct directed graph of rows
    dg = nx.DiGraph()
    dg.add_nodes_from(range(M))
    for n in dg.nodes:
        col_idx = matching[n]
        col_node = col_idx + M
        # For all rows that share this column
        for neighbor in bg[col_node]:
            if neighbor != n:
                # Add an edge towards this column's matched row
                dg.add_edge(neighbor, n)

    # Partition the rows into strongly connected components (diagonal blocks)
    scc_list = list(nxc.strongly_connected_components(dg))
    node_scc_map = {n: idx for idx, scc in enumerate(scc_list) for n in scc}

    # Now we need to put the SCCs in the right order. We do this by performing
    # a topological sort on the DAG of SCCs.
    dag = nx.DiGraph()
    for i, c in enumerate(scc_list):
        dag.add_node(i)
    for n in dg.nodes:
        source_scc = node_scc_map[n]
        for neighbor in dg[n]:
            target_scc = node_scc_map[neighbor]
            if target_scc != source_scc:
                dag.add_edge(target_scc, source_scc)
                # Reverse direction of edge. This corresponds to creating
                # a block lower triangular matrix.

    scc_order = list(nxd.topological_sort(dag))

    scc_block_map = {c: i for i, c in enumerate(scc_order)}
    node_block_map = {n: scc_block_map[c] for n, c in node_scc_map.items()}

    sym_perm_new2old = list(n for i in scc_order for n in scc_list[i])
    sym_perm_old2new = {o: n for n, o in sym_perm_new2old.items()}
    assert len(sym_perm_old2new) == M

    match_col_perm = {c: r for r, c in enumerate(matching)}
    assert len(match_col_perm) == M

    col_perm = [sym_perm_old2new[match_col_perm[j]] for j in range(N)]
    row_perm = sym_perm_old2new

    return row_perm, col_perm
