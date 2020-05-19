"""Community detection code"""

from contrib.community_detection import community_graph
import community


def detect_communities(model, node_type='v', with_objective=True, weighted_graph=True, file_destination=None,
                       random_seed=None):
    """
    Detects communities in a graph of variables and constraints

    This function takes in a Pyomo optimization model, organizes the variables and constraints into a graph of nodes
    and edges, and then uses Louvain community detection to create a dictionary of the communities of the nodes.
    Either variables or constraints can be chosen as the nodes.

    Args:
        model (Block): a Pyomo model or block to be used for community detection
        node_type : a string that specifies the dictionary to be returned; 'v' returns a dictionary with communities
        based on variable nodes, 'c' returns a dictionary with communities based on constraint nodes, and any other
        input returns an error message
        with_objective: a Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified as (see prior argument))
        weighted_graph: a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
        file_destination: an optional argument that takes in a path if the user wants to save an edge and adjacency
        list based on the model
        random_seed : takes in an integer to use as the seed number for the heuristic Louvain community detection

    Returns:
        community_map: a Python dictionary whose keys are integers from zero to the number of communities minus one
        with values that are lists of the nodes in the given community
    """

    if node_type != 'v' and node_type != 'c':
        print("Invalid input: Specify node_type 'v' or 'c' for function detect_communities")
        return None

    # Generate the model_graph (a networkX graph) based on the given Pyomo optimization model
    model_graph = community_graph._generate_model_graph(model, node_type=node_type, with_objective=with_objective,
                                                        weighted_graph=weighted_graph,
                                                        file_destination=file_destination)

    # Use Louvain community detection to determine which community each node belongs to
    partition_of_graph = community.best_partition(model_graph, random_state=random_seed)

    # Use partition_of_graph to create a dictionary that maps communities to nodes (because Louvain community detection
    # returns a dictionary that maps individual nodes to their communities)
    number_of_communities = len(set(partition_of_graph.values()))
    community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)

    return community_map
