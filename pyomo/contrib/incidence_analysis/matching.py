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

from pyomo.common.dependencies import networkx as nx


def maximum_matching(matrix_or_graph, top_nodes=None):
    """Return a maximum cardinality matching of the provided matrix or
    bipartite graph

    If a matrix is provided, the matching is returned as a map from row
    indices to column indices. If a bipartite graph is provided, a list of
    "top nodes" must be provided as well. These correspond to one of the
    "bipartite sets". The matching is then returned as a map from "top nodes"
    to the other set of nodes.

    Parameters
    ----------
    matrix_or_graph: SciPy sparse matrix or NetworkX Graph
        The matrix or graph whose maximum matching will be computed
    top_nodes: list
        Integer nodes representing a bipartite set in a graph. Must be provided
        if and only if a NetworkX Graph is provided.

    Returns
    -------
    max_matching: dict
        Dict mapping from integer nodes in the first bipartite set (row
        indices) to nodes in the second (column indices).

    """
    nxb = nx.algorithms.bipartite
    nxc = nx.algorithms.components
    from_biadjacency_matrix = nxb.matrix.from_biadjacency_matrix

    if isinstance(matrix_or_graph, nx.Graph):
        graph_provided = True
        if top_nodes is None:
            raise RuntimeError("top_nodes argument must be set if a graph is provided.")
        M = len(top_nodes)
        N = len(matrix_or_graph.nodes) - M
        bg = matrix_or_graph
        if not nxb.is_bipartite(bg):
            raise RuntimeError("Provided graph is not bipartite.")
    else:
        graph_provided = False
        # Assume something SciPy-sparse compatible was provided.
        if top_nodes is not None:
            raise RuntimeError(
                "top_nodes argument cannot be used if a matrix is provided"
            )
        M, N = matrix_or_graph.shape
        top_nodes = list(range(M))
        bg = from_biadjacency_matrix(matrix_or_graph)

        # Check assumptions regarding from_biadjacency_matrix function:
        for i in range(M):
            # First M nodes in graph correspond to rows
            assert bg.nodes[i]["bipartite"] == 0

        for j in range(M, M + N):
            # Last N nodes in graph correspond to columns
            assert bg.nodes[j]["bipartite"] == 1

    matching = nxb.maximum_matching(bg, top_nodes=top_nodes)
    if graph_provided:
        top_node_set = set(top_nodes)
        # If a graph was provided, we return a mapping from "top nodes"
        # to their matched "bottom nodes"
        max_matching = {n0: n1 for n0, n1 in matching.items() if n0 in top_node_set}
    else:
        # If a matrix was provided, we return a mapping from row indices
        # to column indices.
        max_matching = {n0: n1 - M for n0, n1 in matching.items() if n0 < M}

    return max_matching
