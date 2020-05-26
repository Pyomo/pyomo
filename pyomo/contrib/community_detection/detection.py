"""
Main module for community detection integration with Pyomo models.

This module separates model variables or constraints into different communities
distinguished by the degree of connectivity between community members.

Original implementation developed by Rahul Joglekar in the Grossmann research group.

"""
from logging import getLogger

from pyomo.common.dependencies import attempt_import
from pyomo.contrib.community_detection.community_graph import _generate_model_graph

logger = getLogger('pyomo.contrib.community_detection')

# Yikes, the naming. I hope the user doesn't have another package installed named 'community'
community, community_available = attempt_import(
    'community', error_message="Could not import the 'community' library, available via 'python-louvain' on PyPI.")


def detect_communities(model, node_type='v', with_objective=True, weighted_graph=True, random_seed=None):
    """
    Detects communities in a Pyomo optimization model

    This function takes in a Pyomo optimization model, organizes the variables and constraints into a graph of nodes
    and edges, and then by using Louvain community detection on the graph, a dictionary is ultimately created, mapping
    the communities to the nodes in each community.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for community detection
    node_type: str
        A string that specifies the dictionary to be returned.
        'v' returns a dictionary with communities based on variable nodes,
        'c' returns a dictionary with communities based on constraint nodes.
    with_objective: bool
        a Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified as (see prior argument))
    weighted_graph: bool
        a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
    random_seed: int, optional
        Specify the integer to use the random seed for the heuristic Louvain community detection

    Returns
    -------
    community_map: dict
        a Python dictionary whose keys are integers from zero to the number of communities minus one
        with values that are sorted lists of the nodes in the given community
    """
    assert node_type in ('v', 'c'), "Invalid node type specified. Valid values: 'v', 'c'."

    # Generate the model_graph (a networkX graph) based on the given Pyomo optimization model
    model_graph = _generate_model_graph(
        model, node_type=node_type, with_objective=with_objective,
        weighted_graph=weighted_graph, )

    # Use Louvain community detection to determine which community each node belongs to
    partition_of_graph = community.best_partition(model_graph, random_state=random_seed)

    # Use partition_of_graph to create a dictionary that maps communities to nodes (because Louvain community detection
    # returns a dictionary that maps individual nodes to their communities)
    number_of_communities = len(set(partition_of_graph.values()))
    community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)

    # Log information about the number of communities found from the model
    logger.info("%s communities were found in the model" % number_of_communities)
    if number_of_communities == 0:
        logger.error("in detect_communities: Empty community map was returned")
    if number_of_communities == 1:
        logger.warning("Community detection found that with the given parameters, the model could not be decomposed - "
                       "only one community was found")

    return community_map


def write_community_to_file(community_map, filename):
    """Writes the community edge and adjacency lists to a file.

    Parameters
    ----------
    community_map
    filename
        an optional argument that takes in a path if the user wants to save an edge and adjacency
        list based on the model

    """
    pass
