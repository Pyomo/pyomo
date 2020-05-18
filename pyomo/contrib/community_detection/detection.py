"""Community detection code"""

from contrib.community_detection import community_graph
import community


def detect_communities(model, node_type='v', with_objective=True, weighted_graph=True, write_to_path=None, random_seed=None):
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
        with_objective: an optional Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified (see prior argument))
        weighted_graph: an optional Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
        write_to_path: an optional argument that takes in a path for edge lists and adjacency lists to be saved
        random_seed : takes in an integer to use as the seed number for the Louvain community detection

    Return:
        community_map: a Python dictionary whose keys are integers from one to the number of communities with
        values that are lists of the nodes in the given community
    """

    if node_type != 'v' and node_type != 'c':
        print("Invalid input: Specify node_type 'v' or 'c' for function detect_communities")
        return None

    model_graph = community_graph._generate_model_graph(model, node_type=node_type, with_objective=with_objective,
                                                        weighted_graph=weighted_graph, write_to_path=write_to_path)

    partition_of_graph = community.best_partition(model_graph, random_state=random_seed)
    n_communities = int(len(set(partition_of_graph.values())))
    community_map = {nth_community: [] for nth_community in range(n_communities)}

    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)

    return community_map

