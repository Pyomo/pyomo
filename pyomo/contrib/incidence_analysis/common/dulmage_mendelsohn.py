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

from pyomo.common.dependencies import networkx_available

if networkx_available:
    from networkx.classes.digraph import DiGraph
    from networkx.algorithms.traversal.breadth_first_search import bfs_edges
    from networkx.algorithms.bipartite.basic import sets as bipartite_sets
    from networkx.algorithms.bipartite.matching import maximum_matching
    from networkx.algorithms.components import connected_components


"""
This module implements the coarse Dulmage-Mendelsohn partition on a Networkx
bipartite graph. It exists in this common directory as it does not depend on
the rest of the package, or Pyomo except through the import of NetworkX.
"""


def _get_projected_digraph(bg, matching, top_nodes):
    digraph = DiGraph()
    digraph.add_nodes_from(top_nodes)
    for n in top_nodes:
        # Add in-edges
        if n in matching:
            for t in bg[matching[n]]:
                if t != n:
                    digraph.add_edge(t, n)
        # Add out-edges
        for b in bg[n]:
            if b in matching and matching[b] != n:
                digraph.add_edge(n, matching[b])
    return digraph


def _get_reachable_from(digraph, sources):
    _filter = set()
    reachable = []
    for node in sources:
        for i, j in bfs_edges(digraph, node):
            if j not in _filter:
                _filter.add(j)
                reachable.append(j)
    return reachable, _filter


def dulmage_mendelsohn(bg, top_nodes=None, matching=None):
    """
    The Dulmage-Mendelsohn decomposition for bipartite graphs.
    This is the coarse decomposition.
    """
    # TODO: Should top_nodes be required? We can try to infer, but
    # the result is in terms of this partition...
    top, bot = bipartite_sets(bg, top_nodes)
    bot_nodes = [n for n in bg if n not in top]
    if top_nodes is None:
        top_nodes = [n for n in bg if n in top]

    if matching is None:
        # This maps top->bot AND bot->top
        matching = maximum_matching(bg, top_nodes=top_nodes)

    t_unmatched = [t for t in top_nodes if t not in matching]
    b_unmatched = [b for b in bot_nodes if b not in matching]

    # A traversal along these graphs corresponds to an alternating path
    t_digraph = _get_projected_digraph(bg, matching, top_nodes)
    b_digraph = _get_projected_digraph(bg, matching, bot_nodes)

    # Nodes reachable by an alternating path from unmatched nodes
    t_reachable, t_filter = _get_reachable_from(t_digraph, t_unmatched)
    b_reachable, b_filter = _get_reachable_from(b_digraph, b_unmatched)

    # Nodes matched with those reachable from unmatched nodes
    t_matched_with_reachable = [matching[b] for b in b_reachable]
    b_matched_with_reachable = [matching[t] for t in t_reachable]

    _filter = t_filter.union(b_filter)
    _filter.update(t_unmatched)
    _filter.update(t_matched_with_reachable)
    _filter.update(b_unmatched)
    _filter.update(b_matched_with_reachable)
    t_other = [t for t in top_nodes if t not in _filter]
    b_other = [matching[t] for t in t_other]

    return (
        (t_unmatched, t_reachable, t_matched_with_reachable, t_other),
        (b_unmatched, b_reachable, b_matched_with_reachable, b_other),
    )
