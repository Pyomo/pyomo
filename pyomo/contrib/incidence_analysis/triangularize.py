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

from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
    # TODO: The fact that we import this function here suggests it should be
    # promoted.
    _get_projected_digraph,
)
from pyomo.common.dependencies import networkx as nx


# TODO: get_scc_dag_of_projection function?
def get_scc_dag_of_projection(graph, top_nodes, matching):
    nxb = nx.algorithms.bipartite
    nxc = nx.algorithms.components
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
    if len(matching) != 2*M:
        raise RuntimeError(
            "get_scc_of_projection does not support bipartite graphs without"
            " a perfect matching. Got a graph with %s nodes per bipartite set"
            " and a matching of cardinality %s." % (M, (len(matching)/2))
        )

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

    # The matching is required to interpret scc_list, so maybe the matching
    # needs to be provided to this function
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
    nxc = nx.algorithms.components
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
    if len(matching) != 2*M:
        raise RuntimeError(
            "get_scc_of_projection does not support bipartite graphs without"
            " a perfect matching. Got a graph with %s nodes per bipartite set"
            " and a matching of cardinality %s." % (M, (len(matching)/2))
        )

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

    scc_order = list(nxd.lexicographical_topological_sort(dag))

    # The "natural" return type, here, is a list of lists. Each inner list
    # is an SCC, and contains tuples of nodes. The "top node", and its matched
    # "bottom node".
    ordered_node_subsets = [
        sorted([(i, matching[i]) for i in scc_list[scc_idx]])
        for scc_idx in scc_order
    ]
    return ordered_node_subsets


def block_triangularize(matrix, matching=None):
    """
    Computes the necessary information to permute a matrix to block-lower
    triangular form, i.e. a partition of rows and columns into an ordered
    set of diagonal blocks in such a permutation.

    Arguments
    ---------
    matrix: A SciPy sparse matrix
    matching: A perfect matching of rows and columns, in the form of a dict
              mapping row indices to column indices

    Returns
    -------
    Two dicts. The first maps each row index to the index of its block in a
    block-lower triangular permutation of the matrix. The second maps each
    column index to the index of its block in a block-lower triangular
    permutation of the matrix.
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
    rc_partition = [[(i, j-M) for i, j in scc] for scc in sccs]
    # Is this the right return value? I think I usually want one list of rows,
    # and one list of columns, not a tuple of (row, column).
    return rc_partition


def map_coords_to_blocks(matrix, matching=None):
    sccs = block_triangularize(matrix, matching=matching)
    row_idx_map = {r: idx for idx, scc in enumerate(sccs) for r, _ in scc}
    col_idx_map = {c: idx for idx, scc in enumerate(sccs) for _, c in scc}
    return row_idx_map, col_idx_map


def get_blocks_from_maps(row_block_map, col_block_map):
    """
    Gets the row and column coordinates of each diagonal block in a
    block triangularization from maps of row/column coordinates to
    block indices.

    Arguments
    ---------
    row_block_map: dict
        Dict mapping each row coordinate to the coordinate of the
        block it belongs to

    col_block_map: dict
        Dict mapping each column coordinate to the coordinate of the
        block it belongs to

    Returns
    -------
    tuple of lists
        The first list is a list-of-lists of row indices that partitions
        the indices into diagonal blocks. The second list is a
        list-of-lists of column indices that partitions the indices into
        diagonal blocks.

    """
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


def get_diagonal_blocks(matrix, matching=None):
    """
    Gets the diagonal blocks of a block triangularization of the provided
    matrix.

    Arguments
    ---------
    coo_matrix
        Matrix to get the diagonal blocks of

    matching
        Dict mapping row indices to column indices in the perfect matching
        to be used by the block triangularization.

    Returns
    -------
    tuple of lists
        The first list is a list-of-lists of row indices that partitions
        the indices into diagonal blocks. The second list is a
        list-of-lists of column indices that partitions the indices into
        diagonal blocks.

    """
    sccs = block_triangularize(matrix, matching=matching)
    row_blocks = [[i for i, _ in scc] for scc in sccs]
    col_blocks = [[j for _, j in scc] for scc in sccs]
    return row_blocks, col_blocks
