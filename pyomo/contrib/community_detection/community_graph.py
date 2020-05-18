"""
@author: Rahul
"""
import os
import networkx as nx
from pyomo.environ import *
from pyomo.core.expr.current import identify_variables
from itertools import combinations


def _generate_model_graph(model, node_type='v', with_objective=True, weighted_graph=True, write_to_path=None):
    """
    Creates a graph of nodes and edges based on a Pyomo optimization model

    This function takes in a Pyomo optimization model, then creates a graphical representation of the model with
     variables as nodes and constraints as edges. Whether or not the objective function is included as a constraint
     equation can be specified, as well as whether or not to create edge and adjacency lists

    Args:
        model (Block): a Pyomo model or block to be used for community detection
        node_type : a string that specifies the node_type of graph to be returned; 'v' returns a graph with variable
        nodes and constraint edges, 'c' returns a graph with constraint nodes and variable edges, and any other input
        returns an error message
        with_objective: an optional Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified (see prior argument))
        weighted_graph: an optional Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
        write_to_path: an optional argument that takes in a path for edge lists and adjacency lists to be saved

    Return:
        model_graph: a weighted or unweighted networkX graph with variable or constraint nodes and edges
    """

    # Detect_communities only calls generate_model_graph with a node_type of 'c' or 'v'
    # Note that if this function is called with a node_type that is not 'c' or 'v', this will execute as if a node
    # type of 'c' was given
    model_graph = nx.Graph()
    edge_weight_dict = dict()
    edge_set = set()

    if node_type == 'v':
        # Create nodes as variables
        for variable in model.component_data_objects(Var, descend_into=True):
            model_graph.add_node(str(variable), variable_name=str(variable))

        # Go through all constraints:
        for constraint in model.component_data_objects(Constraint, descend_into=True):
            # Create a list of the variables that occur in the given constraint equation
            variable_list = [str(variable) for variable in list(identify_variables(constraint.body))]
            edges_between_variables = list(combinations(sorted(variable_list), 2))

            # Update the edge weight dictionary based on this equation
            if weighted_graph:
                edge_weight_dict = _update_edge_weight_dict(edges_between_variables, edge_weight_dict)
            else:
                edge_set.update(set(edges_between_variables))

        if with_objective:
            objective_function = list(model.component_data_objects(Objective, descend_into=True))[0]
            variable_list = [str(variable) for variable in list(identify_variables(objective_function))]
            edges_between_variables = list(combinations(sorted(variable_list), 2))
            if weighted_graph:
                edge_weight_dict = _update_edge_weight_dict(edges_between_variables, edge_weight_dict)
            else:
                edge_set.update(set(edges_between_variables))

    elif node_type == 'c': # Constraint nodes
        # Create nodes as constraints
        for constraint in model.component_data_objects(Constraint, descend_into=True):
            model_graph.add_node(str(constraint), variable_name=str(constraint))

        if with_objective:
            objective_function = list(model.component_data_objects(Objective, descend_into=True))[0]
            model_graph.add_node(str(objective_function), variable_name=str(objective_function))

        # Go through all variables
        for variable in model.component_data_objects(Var, descend_into=True):
            # Create a list of the constraints that occur with the given variable
            constraint_list = [str(constraint) for constraint in
                               model.component_data_objects(Constraint, descend_into=True) if
                               str(variable) in [str(var) for var in identify_variables(constraint.body)]]

            if with_objective and str(variable) in [str(var) for var in list(identify_variables(objective_function))]:
                constraint_list.append(str(objective_function))

            edges_between_constraints = list(combinations(sorted(constraint_list), 2))

            # Update the edge weight dictionary based on this equation
            if weighted_graph:
                edge_weight_dict = _update_edge_weight_dict(edges_between_constraints, edge_weight_dict)
            else:
                edge_set.update(edges_between_constraints)

    else:
        # This case should never get executed because of the way this function is called by detect communities
        print("Node type must be specified as 'v' or 'c' (variable nodes or constraint nodes).")

    if weighted_graph:
        for edge in edge_weight_dict:
            model_graph.add_edge(edge[0], edge[1], weight=edge_weight_dict[edge])
        edge_weight_dict.clear()
    else:
        model_graph.add_edges_from(edge_set)
        edge_set.clear()

    if write_to_path is not None:
        _write_to_file(model_graph, node_type=node_type, with_objective=with_objective,
                       weighted_graph=weighted_graph, write_to_path=write_to_path)

    return model_graph


def _write_to_file(model_graph, node_type, with_objective, weighted_graph, write_to_path):
    """

    :param model_graph:
    :param write_to_path:
    :param node_type:
    :param with_objective:
    """
    community_detection_dir = os.path.join(write_to_path, 'community_detection_graphs')
    if not os.path.exists(community_detection_dir):
        os.makedirs(community_detection_dir)

    if node_type == 'v':
        type_of_node = 'variable'
    else:
        type_of_node = 'constraint'

    if with_objective:
        obj_status = 'with_obj'
    else:
        obj_status = 'without_obj'

    if weighted_graph:
        weight_status = 'weighted'
    else:
        weight_status = 'unweighted'

    nx.write_edgelist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                      '.%s_%s_edge_list_%s' % (type_of_node, weight_status, obj_status))
    nx.write_adjlist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                     '.%s_%s_adj_list_%s' % (type_of_node, weight_status, obj_status))


def _update_edge_weight_dict(edge_list, edge_weight_dict):
    """
    Updates a dictionary of edge weights given a list of edges

    This function takes in a list of edges on a graph and an existing dictionary that maps edges to weights. Then,
    using the edge list, the dictionary of edge weights is updated and returned

    Args:
        edge_list : a Python list containing a list of nodes in tuples (two nodes in a tuple indicate an edge that
        needs to be drawn)
        edge_weight_dict : a Python dictionary containing all of the existing edges to be drawn on the graph mapped to
        their weights (an edge that occurs n times has a weight of n)

    Return:
        edge_weight_dict : a Python dictionary containing all of the existing edges to be drawn on the graph mapped to
        their weights (an edge that occurs n times has a weight of n); when it is returned, it has been updated with
        the edges in edge_list
    """
    for edge in edge_list:
        if edge not in edge_weight_dict:
            edge_weight_dict[edge] = 1
        else:
            edge_weight_dict[edge] += 1
    return edge_weight_dict
