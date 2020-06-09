"""
Main module for community detection integration with Pyomo models.

This module separates model variables or constraints into different communities
distinguished by the degree of connectivity between community members.

Original implementation developed by Rahul Joglekar in the Grossmann research group.

"""
from logging import getLogger

from pyomo.common.dependencies import attempt_import
from pyomo.core import ConcreteModel, Var
from pyomo.contrib.community_detection.community_graph import _generate_model_graph
from pyomo.common.dependencies import networkx as nx

import os

logger = getLogger('pyomo.contrib.community_detection')

# Attempt import of louvain community detection package
community_louvain, community_louvain_available = attempt_import(
    'community', error_message="Could not import the 'community' library, available via 'python-louvain' on PyPI.")


def detect_communities(model, node_type='c', with_objective=True, weighted_graph=True, random_seed=None,
                       string_output=False):
    """
    Detects communities in a Pyomo optimization model

    This function takes in a Pyomo optimization model, organizes the variables and constraints into a graph of nodes
    and edges, and then by using Louvain community detection on the graph, a dictionary is ultimately created, mapping
    the communities to the nodes in each community.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for community detection
    node_type: str, optional
        A string that specifies the dictionary to be returned, the default is 'c'.
        'c' returns a dictionary with communities based on constraint nodes,
        'v' returns a dictionary with communities based on variable nodes,
        'b' returns a dictionary with communities based on constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function will be
        included as a node/constraint (depending on what node_type is specified as (see prior argument)), the default
        is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be created from the Pyomo
        model; the default is True (node_type='b' creates an unweighted graph regardless of this parameter)
    random_seed: int, optional
        An integer that is used as the random seed for the heuristic Louvain community detection
    string_output: bool, optional
        a Boolean argument that specifies whether the community map that is returned contains communities of the
        strings of the nodes or if it contains the actual Pyomo modeling components (the default is False)

    Returns
    -------
    community_map: dict
        a Python dictionary whose keys are integers from zero to the number of communities minus one
        with values that are sorted lists of the nodes in the given community
    """

    # Check that all arguments are of the correct type
    assert isinstance(model, ConcreteModel), "Invalid model: 'model=%s' - model must be an instance of " \
                                             "ConcreteModel" % model

    assert node_type in ('b', 'c', 'v'), "Invalid node type specified: 'node_type=%s' - Valid " \
                                         "values: 'c', 'v', 'b'" % node_type

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert type(random_seed) == int or random_seed is None, "Invalid value for random_seed: 'random_seed=%s' - " \
                                                            "random_seed must be a non-negative integer" % random_seed

    assert type(string_output) == bool, "Invalid value for with_objective: 'string_output=%s' - string_output " \
                                        "must be a Boolean" % string_output

    # Generate the model_graph (a networkX graph) based on the given Pyomo optimization model
    # string_map maps the string of Pyomo modeling components to the actual components themselves
    # constraint_variable_map maps a constraint to the variables it contains
    model_graph, string_map, constraint_variable_map = _generate_model_graph(
        model, node_type=node_type, with_objective=with_objective,
        weighted_graph=weighted_graph)

    # Use Louvain community detection - this returns a dictionary mapping individual nodes to their communities
    partition_of_graph = community_louvain.best_partition(model_graph, random_state=random_seed)

    # Now, use partition_of_graph to create a dictionary (str_community_map) that maps communities to their nodes
    number_of_communities = len(set(partition_of_graph.values()))
    str_community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        str_community_map[nth_community].append(node)

    # At this point, we have str_community_map, which maps an integer (the community number) to a list of the strings
    # of the Pyomo modeling components in each community

    # Now, we want to include another list (so that each key in str_community_map corresponds to a tuple of two lists),
    # which wil be determined based on the node_type given by the user

    # Constraint node type - for a given community, we want to create a second list that contains all of the variables
    # contained in the given constraints
    if node_type == 'c':
        for community_key in str_community_map:
            main_list = str_community_map[community_key]
            variable_list = []
            for str_constraint in main_list:
                variable_list.extend(constraint_variable_map[str_constraint])
            variable_list = sorted(set(variable_list))
            str_community_map[community_key] = (main_list, variable_list)

    # Variable node type - for a given community, we want to create a second list that contains all of the constraints
    # that the variables appear in
    elif node_type == 'v':
        for community_key in str_community_map:
            main_list = str_community_map[community_key]
            constraint_list = []
            for str_variable in main_list:
                constraint_list.extend([constraint_key for constraint_key in constraint_variable_map if
                                        str_variable in constraint_variable_map[constraint_key]])
            constraint_list = sorted(set(constraint_list))
            str_community_map[community_key] = (main_list, constraint_list)

    # Both variable and constraint nodes (bipartite graph) - for a given community, we simply want to separate the
    # nodes into their two groups; thus, we create a list of constraints and a list of variables
    elif node_type == 'b':
        for community_key in str_community_map:
            constraint_node_list, variable_node_list = [], []
            node_community_list = str_community_map[community_key]
            for str_node in node_community_list:
                node = string_map[str_node]
                if isinstance(node, Var):
                    variable_node_list.append(str_node)
                else:
                    constraint_node_list.append(str_node)
            str_community_map[community_key] = (constraint_node_list, variable_node_list)

    # Log information about the number of communities found from the model
    logger.info("%s communities were found in the model" % number_of_communities)
    if number_of_communities == 0:
        logger.error("in detect_communities: Empty community map was returned")
    if number_of_communities == 1:
        logger.warning("Community detection found that with the given parameters, the model could not be decomposed - "
                       "only one community was found")

    # If the user desires an easy-to-read output, setting string_output to True will return the str_community_map
    # that we have right now
    if string_output:
        return str_community_map

    # If string_output is not set to True, then we will go ahead and convert the strings in the communities
    # to the actual Pyomo modeling components that they correspond to

    # Now, using string_map, we will convert str_community_map into community_map, a dictionary of the actual
    # variables/constraints/objectives
    community_map = {}
    for nth_community in str_community_map:
        first_list = str_community_map[nth_community][0]
        second_list = str_community_map[nth_community][1]
        new_first_list = [string_map[community_member] for community_member in first_list]
        new_second_list = [string_map[community_member] for community_member in second_list]
        community_map[nth_community] = (new_first_list, new_second_list)

    # Return community_map, which has integer keys that now map to a tuple of two lists
    # containing Pyomo modeling components
    return community_map


def get_edge_list(model, node_type='c', with_objective=True, weighted_graph=True, file_path=None):
    """
    Creates an edge list from on a given Pyomo optimization model

    This function takes in a Pyomo optimization model, creates a networkX graph based on that model, then
    returns an edge list based on the networkX graph. If the user provides a file path, then the edge list will
    also be saved in a new directory.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for generating an edge list
    node_type: str, optional
        A string that specifies the type of graph that is created from the model, the default is 'c'.
        'c' creates a graph based on constraint nodes,
        'v' creates a graph based on variable nodes,
        'b' creates a graph based on constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function will be
        included as a node/constraint (depending on what node_type is specified as (see prior argument)), the default
        is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model (the default is True)
    file_path: str, optional
        a string that specifies a path if the user wants to write the edge list to a file

    Returns
    -------
    edge_list: generator
        a networkX edge list created from a networkX graph based on a given Pyomo optimization model
    """
    # Check that all arguments are of the correct type
    assert isinstance(model, ConcreteModel), "Invalid model: 'model=%s' - model must be an instance of " \
                                             "ConcreteModel" % model

    assert node_type in ('c', 'v', 'b'), "Invalid node type specified: 'node_type=%s' - Valid " \
                                         "values: 'c', 'v', 'b'" % node_type

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert type(file_path) == str, "Invalid file path given: 'file_path=%s' - file_path must be a string" % file_path

    model_graph = _generate_model_graph(model, node_type=node_type, with_objective=with_objective,
                                        weighted_graph=weighted_graph)[0]

    # If no file path is given by the user, then the edge list is generated and immediately returned to the user
    edge_list = nx.generate_edgelist(model_graph)
    if file_path is None:
        return edge_list

    # Create a path (community_detection_dir) that joins the user-provided file_destination with the
    # directory where we will save the edge list (community_detection_graph_info)
    community_detection_dir = os.path.join(file_path, 'community_detection_graph_info')

    # In case the user-provided file_destination does not exist, we will use os.makedirs to create
    # intermediate directories so that community_detection_dir is now a valid path and log this as a warning
    if not os.path.exists(community_detection_dir):
        os.makedirs(community_detection_dir)
        logger.warning("in detect_communities: The given file path did not exist so the following file path was "
                       "created and used to store the edge list: %s" % community_detection_dir)

    # Collect information for naming the edge list:

    # Based on node_type, determine the type of node
    if node_type == 'c':
        type_of_node = 'constraint'
    elif node_type == 'v':
        type_of_node = 'variable'
    else:
        type_of_node = 'bipartite'

    # Based on whether the objective function was included in creating the model graph, determine objective status
    if with_objective:
        obj_status = 'with_obj'
    else:
        obj_status = 'without_obj'

    # Based on whether the model graph was weighted or unweighted, determine weight status
    if weighted_graph:
        weight_status = 'weighted'
    else:
        weight_status = 'unweighted'

    # Now, using all of this information, use the networkX function to write the edge list to the
    # file path determined above and name it using the relevant graph information organized above
    nx.write_edgelist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                      '.%s_%s_edge_list_%s' % (type_of_node, weight_status, obj_status))

    # Now, return the edge list
    return edge_list


def get_adj_list(model, node_type='c', with_objective=True, weighted_graph=True, file_path=None):
    """
    Creates an edge list from on a given Pyomo optimization model

    This function takes in a Pyomo optimization model, creates a networkX graph based on that model, then
    returns an adjacency list based on the networkX graph. If the user provides a file path, then the edge list will
    also be saved in a new directory.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for generating an adjacency list
    node_type: str, optional
        A string that specifies the type of graph that is created from the model, the default is 'c'.
        'c' creates a graph based on constraint nodes,
        'v' creates a graph based on variable nodes,
        'b' creates a graph based on constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function will be
        included as a node/constraint (depending on what node_type is specified as (see prior argument)), the default
        is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model (the default is True)
    file_path: str, optional
        a string that specifies a path if the user wants to write the adjacency list to a file

    Returns
    -------
    adj_list: generator
        a networkX adjacency list created from a networkX graph based on a given Pyomo optimization model
    """
    # Check that all arguments are of the correct type
    assert isinstance(model, ConcreteModel), "Invalid model: 'model=%s' - model must be an instance of " \
                                             "ConcreteModel" % model

    assert node_type in ('c', 'v', 'b'), "Invalid node type specified: 'node_type=%s' - Valid " \
                                         "values: 'c', 'v', 'b'" % node_type

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert type(file_path) == str, "Invalid file path given: 'file_path=%s' - file_path must be a string" % file_path

    model_graph = _generate_model_graph(model, node_type=node_type, with_objective=with_objective,
                                        weighted_graph=weighted_graph)[0]

    # If no file path is given by the user, then the adjacency list is generated and immediately returned to the user
    adj_list = nx.generate_adjlist(model_graph)
    if file_path is None:
        return adj_list

    # Create a path (community_detection_dir) that joins the user-provided file_destination with the
    # directory where we will save the adjacency list (community_detection_graph_info)
    community_detection_dir = os.path.join(file_path, 'community_detection_graph_info')

    # In case the user-provided file_destination does not exist, we will use os.makedirs to create
    # intermediate directories so that community_detection_dir is now a valid path and log this as a warning
    if not os.path.exists(community_detection_dir):
        os.makedirs(community_detection_dir)
        logger.warning("in detect_communities: The given file path did not exist so the following file path was "
                       "created and used to store the adjacency list: %s" % community_detection_dir)

    # Collect information for naming the adjacency list:

    # Based on node_type, determine the type of node
    if node_type == 'c':
        type_of_node = 'constraint'
    elif node_type == 'v':
        type_of_node = 'variable'
    else:
        type_of_node = 'bipartite'

    # Based on whether the objective function was included in creating the model graph, determine objective status
    if with_objective:
        obj_status = 'with_obj'
    else:
        obj_status = 'without_obj'

    # Based on whether the model graph was weighted or unweighted, determine weight status
    if weighted_graph:
        weight_status = 'weighted'
    else:
        weight_status = 'unweighted'

    # Now, using all of this information, use the networkX function to write the adjacency list to the
    # file path determined above and name it using the relevant graph information organized above
    nx.write_adjlist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                     '.%s_%s_adj_list_%s' % (type_of_node, weight_status, obj_status))

    # Now, return the adjacency list
    return adj_list
