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

from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
    # TODO: The fact that we import this function here suggests it should be
    # promoted.
    _get_projected_digraph,
)
from pyomo.common.dependencies import networkx as nx


def _get_scc_dag_of_projection(graph, top_nodes, matching):
    """Return the DAG of strongly connected components of a bipartite graph,
    projected with respect to a perfect matching

    This data structure can be used, for instance, to identify the minimal
    subsystem of constraints and variables necessary to solve a given variable
    or constraint.

    """
    nxc = nx.algorithms.components
    # _get_projected_digraph treats matched edges as "in-edges", so we
    # reverse the direction of edges here.
    dg = _get_projected_digraph(graph, matching, top_nodes).reverse()

    scc_list = list(nxc.strongly_connected_components(dg))
    n_scc = len(scc_list)
    node_scc_map = {n: idx for idx, scc in enumerate(scc_list) for n in scc}

    # Now we need to put the SCCs in the right order. We do this by performing
    # a topological sort on the DAG of SCCs.
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_scc))
    for n in dg.nodes:
        source_scc = node_scc_map[n]
        for neighbor in dg[n]:
            target_scc = node_scc_map[neighbor]
            if target_scc != source_scc:
                dag.add_edge(source_scc, target_scc)

    # Note that the matching is required to fully interpret scc_list (as it
    # only contains the "top nodes")
    return scc_list, dag


def get_scc_of_projection(graph, top_nodes, matching=None):
    """Return the topologically ordered strongly connected components of a
    bipartite graph, projected with respect to a perfect matching

    The provided undirected bipartite graph is projected into a directed graph
    on the set of "top nodes" by treating "matched edges" as out-edges and
    "unmatched edges" as in-edges. Then the strongly connected components of
    the directed graph are computed. These strongly connected components are
    unique, regardless of the choice of perfect matching. The strongly connected
    components form a directed acyclic graph, and are returned in a topological
    order. The order is unique, as ambiguities are resolved "lexicographically".

    The "direction" of the projection (where matched edges are out-edges)
    leads to a block *lower* triangular permutation when the top nodes
    correspond to *rows* in the bipartite graph of a matrix.

    Parameters
    ----------
    graph: NetworkX Graph
        A bipartite graph
    top_nodes: list
        One of the bipartite sets in the graph
    matching: dict
        Maps each node in ``top_nodes`` to its matched node

    Returns
    -------
    list of lists
        The outer list is a list of strongly connected components. Each
        strongly connected component is a list of tuples of matched nodes.
        The first node is a "top node", and the second is an "other node".

    """
    nxb = nx.algorithms.bipartite
    nxd = nx.algorithms.dag
    if not nxb.is_bipartite(graph):
        raise RuntimeError("Provided graph is not bipartite.")
    M = len(top_nodes)
    N = len(graph.nodes) - M
    if M != N:
        raise RuntimeError(
            "get_scc_of_projection does not support bipartite graphs with"
            " bipartite sets of different cardinalities. Got sizes %s and"
            " %s." % (M, N)
        )
    if matching is None:
        # This matching maps top nodes to "other nodes" *and* other nodes
        # back to top nodes.
        matching = nxb.maximum_matching(graph, top_nodes=top_nodes)
    if len(matching) != 2 * M:
        raise RuntimeError(
            "get_scc_of_projection does not support bipartite graphs without"
            " a perfect matching. Got a graph with %s nodes per bipartite set"
            " and a matching of cardinality %s." % (M, (len(matching) / 2))
        )

    scc_list, dag = _get_scc_dag_of_projection(graph, top_nodes, matching)
    scc_order = list(nxd.lexicographical_topological_sort(dag))

    # The "natural" return type, here, is a list of lists. Each inner list
    # is an SCC, and contains tuples of nodes. The "top node", and its matched
    # "bottom node".
    ordered_node_subsets = [
        sorted([(i, matching[i]) for i in scc_list[scc_idx]]) for scc_idx in scc_order
    ]
    return ordered_node_subsets


def block_triangularize(matrix, matching=None):
    """Compute ordered partitions of the matrix's rows and columns that
    permute the matrix to block lower triangular form

    Subsets in the partition correspond to diagonal blocks in the block
    triangularization. The order is topological, with ties broken
    "lexicographically".

    Parameters
    ----------
    matrix: ``scipy.sparse.coo_matrix``
        Matrix whose rows and columns will be permuted
    matching: ``dict``
        A perfect matching. Maps rows to columns *and* columns back to rows.

    Returns
    -------
    row_partition: list of lists
        A partition of rows. The inner lists hold integer row coordinates.
    col_partition: list of lists
        A partition of columns. The inner lists hold integer column coordinates.


    .. note::

       **Breaking change in Pyomo 6.5.0**

       The pre-6.5.0 ``block_triangularize`` function returned maps from
       each row or column to the index of its block in a block
       lower triangularization as the original intent of this function
       was to identify when coordinates do or don't share a diagonal block
       in this partition. Since then, the dominant use case of
       ``block_triangularize`` has been to partition variables and
       constraints into these blocks and inspect or solve each block
       individually. A natural return type for this functionality is the
       ordered partition of rows and columns, as lists of lists.
       This functionality was previously available via the
       ``get_diagonal_blocks`` method, which was confusing as it did not
       capture that the partition was the diagonal of a block
       *triangularization* (as opposed to diagonalization). The pre-6.5.0
       functionality of ``block_triangularize`` is still available via the
       ``map_coords_to_block_triangular_indices`` function.

    """
    nxb = nx.algorithms.bipartite
    nxc = nx.algorithms.components
    nxd = nx.algorithms.dag
    from_biadjacency_matrix = nxb.matrix.from_biadjacency_matrix
    M, N = matrix.shape
    if M != N:
        raise ValueError(
            "block_triangularize does not currently support non-square"
            " matrices. Got matrix with shape %s." % ((M, N),)
        )
    graph = from_biadjacency_matrix(matrix)
    row_nodes = list(range(M))
    sccs = get_scc_of_projection(graph, row_nodes, matching=matching)
    row_partition = [[i for i, j in scc] for scc in sccs]
    col_partition = [[j - M for i, j in scc] for scc in sccs]
    return row_partition, col_partition


def map_coords_to_block_triangular_indices(matrix, matching=None):
    row_blocks, col_blocks = block_triangularize(matrix, matching=matching)
    row_idx_map = {r: idx for idx, rblock in enumerate(row_blocks) for r in rblock}
    col_idx_map = {c: idx for idx, cblock in enumerate(col_blocks) for c in cblock}
    return row_idx_map, col_idx_map


@deprecated(
    msg=(
        "``get_blocks_from_maps`` is deprecated. This functionality has been"
        " incorporated into ``block_triangularize``."
    ),
    version="6.5.0",
)
def get_blocks_from_maps(row_block_map, col_block_map):
    blocks = set(row_block_map.values())
    assert blocks == set(col_block_map.values())
    n_blocks = len(blocks)
    block_rows = [[] for _ in range(n_blocks)]
    block_cols = [[] for _ in range(n_blocks)]
    for r, b in row_block_map.items():
        block_rows[b].append(r)
    for c, b in col_block_map.items():
        block_cols[b].append(c)
    return block_rows, block_cols


@deprecated(
    msg=(
        "``get_diagonal_blocks`` has been deprecated. Please use"
        " ``block_triangularize`` instead."
    ),
    version="6.5.0",
)
def get_diagonal_blocks(matrix, matching=None):
    return block_triangularize(matrix, matching=matching)
