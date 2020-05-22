"""Community Detection Code - Rahul Joglekar"""
from pyomo.core import ConcreteModel
from contrib.community_detection import community_graph
import community
import logging


def detect_communities(model, node_type='v', with_objective=True, weighted_graph=True, file_destination=None,
                       log_level=logging.WARNING, random_seed=None):
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
        log_level: determines the minimum severity of an event for it to be included in the event logger file; can be
        specified as any of the following (in order of increasing severity): logging.DEBUG, logging.INFO,
        logging.WARNING, logging.ERROR, logging.CRITICAL; These levels correspond to integer values of 10, 20, 30, 40,
        and 50 (respectively). Thus, log_level can also be specified as an integer.
        random_seed : takes in an integer to use as the seed number for the heuristic Louvain community detection

    Returns:
        community_map: a Python dictionary whose keys are integers from zero to the number of communities minus one
        with values that are lists of the nodes in the given community
    """

    # Use this function as a check to make sure all of the arguments are of the correct type, else return None
    if check_for_correct_arguments(model, node_type, with_objective, weighted_graph, file_destination, log_level,
                                   random_seed) is False:
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

    # Log information about the number of communities found from the model
    logging.info("%s communities were found in the model" % number_of_communities)
    if number_of_communities == 0:
        logging.error("in detect_communities: Empty community map was returned")
    if number_of_communities == 1:
        logging.warning("Community detection found that with the given parameters, the model could not be decomposed - "
                        "only one community was found")

    return community_map


def check_for_correct_arguments(model, node_type, with_objective, weighted_graph, file_destination, log_level,
                                random_seed):
    """
    Determines whether the arguments given are of the correct types

    This function takes in the arguments given to the function detect_communities and tests whether or not all of them
    are of the correct type. If they are not, the function detect_communities will return None and the incorrect
    arguments will be logged by the event logger as errors.

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
        log_level: determines the minimum severity of an event for it to be included in the event logger file; can be
        specified as any of the following (in order of increasing severity): logging.DEBUG, logging.INFO,
        logging.WARNING, logging.ERROR, logging.CRITICAL; These levels correspond to integer values of 10, 20, 30, 40,
        and 50 (respectively). Thus, log_level can also be specified as an integer.
        random_seed : takes in an integer to use as the seed number for the heuristic Louvain community detection

    Returns:
        correct_arguments: a Boolean that indicates whether the arguments are of the correct type
    """

    # Assume the given arguments are all of the correct type and set this indicator variable to True
    correct_arguments = True

    # Check log_level
    if not isinstance(log_level, int):
        # Configure logger so that the error message is properly formatted
        logging.basicConfig(filename='community_detection_event_log.log', format='%(levelname)s:%(message)s',
                            filemode='w', level=logging.WARNING)
        logging.error(" Invalid argument for function detect_communities: 'log_level=%s' (log_level must be "
                      "of type int)" % log_level)
        correct_arguments = False

    # If the log_level is an int, then configure the logger as specified by the user
    else:
        logging.basicConfig(filename='community_detection_event_log.log', format='%(levelname)s:%(message)s',
                            filemode='w', level=log_level)

    # Check that model is a ConcreteModel
    if not isinstance(model, ConcreteModel):
        logging.error(" Invalid argument for function detect_communities: 'model=%s' (model must be of type "
                      "ConcreteModel)" % model)
        correct_arguments = False

    # Check node_type
    if node_type != 'v' and node_type != 'c':
        logging.error(" Invalid argument for function detect_communities: 'node_type=%s' (node_type must be "
                      "'v' or 'c')" % node_type)
        correct_arguments = False

    # Check with_objective
    if not isinstance(with_objective, bool):
        logging.error(" Invalid argument for function detect_communities: 'with_objective=%s' (weighted_graph must be "
                      "a Boolean)" % with_objective)
        correct_arguments = False

    # Check weighted_graph
    if not isinstance(weighted_graph, bool):
        logging.error(" Invalid argument for function detect_communities: 'weighted_graph=%s' (with_objective must be "
                      "a Boolean)" % weighted_graph)
        correct_arguments = False

    # Check file_destination
    if file_destination is not None and not isinstance(file_destination, str):
        logging.error(" Invalid argument for function detect_communities: 'file_destination=%s' (file_destination must "
                      "be a string)" % file_destination)
        correct_arguments = False

    # Check random_seed
    if random_seed is not None and not isinstance(random_seed, int):
        logging.error(" Invalid argument for function detect_communities: 'random_seed=%s' (random_seed must be "
                      "of type int)" % random_seed)
        correct_arguments = False

    # At this point, if any arguments were not of the correct type, then correct_arguments will be False; if all of the
    # arguments are of the correct type, then correct_arguments will be true
    return correct_arguments
