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

import os
import networkx as nx

logger = getLogger('pyomo.contrib.community_detection')

# Attempt import of louvain community detection package
community, community_available = attempt_import(
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
        A string that specifies the dictionary to be returned.
        'c' returns a dictionary with communities based on constraint nodes,
        'v' returns a dictionary with communities based on variable nodes,
        'b' returns a dictionary with communities based on constraint and variable nodes.
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified as (see prior argument))
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
    random_seed: int, optional
        Specify the integer to use the random seed for the heuristic Louvain community detection
    string_output: bool, optional
        a Boolean argument that specifies whether the community map that is returned is comprised of the strings
        of the nodes or if it contains the actual objects themselves (Pyomo variables/constraints)

    Returns
    -------
    community_map: dict
        a Python dictionary whose keys are integers from zero to the number of communities minus one
        with values that are sorted lists of the nodes in the given community
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

    assert type(random_seed) == int or random_seed is None, "Invalid value for random_seed: 'random_seed=%s' - " \
                                                            "random_seed must be a non-negative integer" % random_seed

    assert type(string_output) == bool, "Invalid value for with_objective: 'string_output=%s' - string_output " \
                                        "must be a Boolean" % string_output

    # Generate the model_graph (a networkX graph) based on the given Pyomo optimization model
    model_graph, string_map, constraint_variable_map = _generate_model_graph(
        model, node_type=node_type, with_objective=with_objective,
        weighted_graph=weighted_graph)

    # Use Louvain community detection to determine which community each node belongs to
    partition_of_graph = community.best_partition(model_graph, random_state=random_seed)

    # Use partition_of_graph to create a dictionary that maps communities to nodes (because Louvain community detection
    # returns a dictionary that maps individual nodes to their communities)
    number_of_communities = len(set(partition_of_graph.values()))
    str_community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        str_community_map[nth_community].append(node)

    # Node type 'c'
    if node_type == 'c':
        for community_key in str_community_map:
            main_list = str_community_map[community_key]
            variable_list = []
            for str_constraint in main_list:
                variable_list.extend(constraint_variable_map[str_constraint])
            variable_list = sorted(set(variable_list))
            str_community_map[community_key] = (main_list, variable_list)

    # Node type 'v'
    elif node_type == 'v':
        for community_key in str_community_map:
            main_list = str_community_map[community_key]
            constraint_list = []
            for str_variable in main_list:
                constraint_list.extend([constraint_key for constraint_key in constraint_variable_map if
                                        str_variable in constraint_variable_map[constraint_key]])
            constraint_list = sorted(set(constraint_list))
            str_community_map[community_key] = (main_list, constraint_list)

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

    if string_output:
        return str_community_map

    # Convert str_community_map into a dictionary of the actual variables/constraints so that it can be iterated over
    # if desired
    community_map = {}
    for nth_community in str_community_map:
        first_list = str_community_map[nth_community][0]
        second_list = str_community_map[nth_community][1]
        new_first_list = [string_map[community_member] for community_member in first_list]
        new_second_list = [string_map[community_member] for community_member in second_list]
        community_map[nth_community] = (new_first_list, new_second_list)

    return community_map


def get_edge_list(model, node_type='c', with_objective=True, weighted_graph=True, file_path=None):
    """Writes the community edge list to a file.

    Parameters
    ----------
    model
    node_type
    with_objective
    weighted_graph
    file_path

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

    if file_path is None:
        edge_list = nx.generate_edgelist(model_graph)
        return edge_list

    else:
        # Create a path based on the user-provided file_destination and the directory where the function will store the
        # edge list (community_detection_graph_info)
        community_detection_dir = os.path.join(file_path, 'community_detection_graph_info')

        # In case the user-provided file_destination does not exist, create intermediate directories so that
        # community_detection_dir is now a valid path
        if not os.path.exists(community_detection_dir):
            os.makedirs(community_detection_dir)
            logger.error("in detect_communities: The given file path did not exist so the following file path was "
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

        # Now, using all of this information, use the networkX functions to write the edge list to the
        # file path determined above and name them using the relevant graph information organized above
        nx.write_edgelist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                          '.%s_%s_edge_list_%s' % (type_of_node, weight_status, obj_status))


def get_adj_list(model, node_type='c', with_objective=True, weighted_graph=True, file_path=None):
    """Writes the community adjacency list to a file.

    Parameters
    ----------
    model
    node_type
    with_objective
    weighted_graph
    file_path

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

    if file_path is None:
        edge_list = nx.generate_adjlist(model_graph)
        return edge_list

    else:
        # Create a path based on the user-provided file_destination and the directory where the function will store the
        # adjacency list (community_detection_graph_info)
        community_detection_dir = os.path.join(file_path, 'community_detection_graph_info')

        # In case the user-provided file_destination does not exist, create intermediate directories so that
        # community_detection_dir is now a valid path
        if not os.path.exists(community_detection_dir):
            os.makedirs(community_detection_dir)
            logger.error("in detect_communities: The given file path did not exist so the following file path was "
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

        # Now, using all of this information, use the networkX functions to write the adjacency list to the
        # file path determined above and name them using the relevant graph information organized above
        nx.write_adjlist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                         '.%s_%s_adj_list_%s' % (type_of_node, weight_status, obj_status))
