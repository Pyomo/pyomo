"""Model Graph Generator Code"""

from core.kernel.component_set import ComponentSet
from pyomo.common.dependencies import networkx as nx
from pyomo.core import Constraint, Objective, Var, ComponentMap
from pyomo.core.expr.current import identify_variables
from itertools import combinations
import os
import logging


def _generate_model_graph(model, node_type='v', with_objective=True, weighted_graph=True, file_destination=None):
    """
    Creates a networkX graph of nodes and edges based on a Pyomo optimization model

    This function takes in a Pyomo optimization model, then creates a graphical representation of the model with
    specific features of the graph determined by the user; the default graph is weighted, has variable nodes, and treats
    the objective function as a constraint in the graph.
    If the user chooses variable nodes, then the edge between two given nodes is created if those two variables occur
    together in the same constraint equation. The weight of each edge depends on the number of constraint equations
    in which the two variables occur together.
    If the user chooses constraint nodes, then the edge between two given nodes is created if those two constraint
    equations share a common variable. The weight of each edge depends on the number of variables common to the two
    constraint equations.
    This function is designed to be called by detect_communities.

    Args:
        model (Block): a Pyomo model or block to be used for community detection
        node_type : a string that specifies the node_type of the graph; node_type='v' creates a graph with variable
        nodes and constraint edges. node_type='c' returns a graph with constraint nodes and variable edges, and any
        other input returns an error message
        with_objective: a Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified as (see prior argument))
        weighted_graph: a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
        file_destination: an optional argument that takes in a path if the user wants to save an edge and adjacency
        list based on the model

    Returns:
        model_graph: a networkX graph with nodes and edges based on the given Pyomo optimization model
    """

    model_graph = nx.Graph()
    if weighted_graph:
        edge_weight_dict = ComponentMap()
    else:
        edge_set = ComponentSet()

    # Variable nodes
    if node_type == 'v':
        # Create nodes based on variables in the Pyomo model
        for model_variable in model.component_data_objects(Var, descend_into=True):
            model_graph.add_node(id(model_variable), node_name=str(model_variable))

        # Loop through all constraints in the Pyomo model
        for model_constraint in model.component_data_objects(Constraint, descend_into=True):
            # Create a set of the variables that occur in the given constraint equation
            variables_in_constraint_equation = ComponentSet(identify_variables(model_constraint.body))

            # Create a list of all the edges that need to be created based on this constraint equation
            edges_between_nodes = list(combinations(variables_in_constraint_equation, 2))

            # Update edge_weight_dict or edge_set based on the determined edges_between_nodes
            if weighted_graph:
                new_edge_weights = ComponentMap(
                    (edge, edge_weight_dict.get(edge, 0) + 1) for edge in edges_between_nodes)
                edge_weight_dict.update(new_edge_weights)
            else:
                new_edges = ComponentSet(edges_between_nodes)
                edge_set.update(new_edges)

        # This if statement will be executed if the user chooses to treat the objective function as a constraint in
        # this model graph
        if with_objective:
            # Use a for loop to account for the possibility of multiple objective functions
            for objective_function in model.component_data_objects(Objective, descend_into=True):

                # Create a list of the variables that occur in the given objective function
                variables_in_objective_function = ComponentSet(identify_variables(objective_function))

                # Create a list of all the edges that need to be created based on the variables in the objective
                edges_between_nodes = list(combinations(variables_in_objective_function, 2))

                # Update edge_weight_dict or edge_set based on the determined edges_between_nodes
                if weighted_graph:
                    new_edge_weights = ComponentMap(
                        (edge, edge_weight_dict.get(edge, 0) + 1) for edge in edges_between_nodes)
                    edge_weight_dict.update(new_edge_weights)
                else:
                    new_edges = ComponentSet(edges_between_nodes)
                    edge_set.update(new_edges)

    # Constraint nodes
    elif node_type == 'c':
        # Create nodes based on constraints in the Pyomo model
        for model_constraint in model.component_data_objects(Constraint, descend_into=True):
            model_graph.add_node(id(model_constraint), node_name=str(model_constraint))

        # If the user chooses to include the objective function as a constraint in the model graph
        if with_objective:
            # Use a loop to account for the possibility of multiple objective functions
            for objective_function in model.component_data_objects(Objective, descend_into=True):
                # Add objective_function as a node in model_graph
                model_graph.add_node(id(objective_function), node_name=str(objective_function))

        # Loop through all variables in the Pyomo model
        for model_variable in model.component_data_objects(Var, descend_into=True):
            # Create a list of the constraint equations that contain the given variable

            # This list comprehension is saying to add a constraint to the list constraints_with_given_variable only
            # if the given model_variable occurs in that constraint
            constraints_with_given_variable = ComponentSet(
                model_constraint for model_constraint in
                model.component_data_objects(Constraint, descend_into=True) if
                model_variable in ComponentSet(identify_variables(model_constraint.body)))

            # Now, if the user is including the objective function as a constraint in the graph, we will add the
            # objective function to the list constraints_with_given_variable only if the given model_variable occurs
            # in the given objective function
            if with_objective:
                # Though this looks confusing, it is structurally and logically identical to the list comprehension
                # used to create a list of the constraints that contain the given model_variable

                # This list comprehension is just saying that the objective function will be added to this list
                # if the model_variable occurs in the given objective_function and the structure reflects the
                # possibility of multiple objective functions
                objective_functions_with_given_variable = ComponentSet(
                    objective_function for objective_function in
                    model.component_data_objects(Objective, descend_into=True) if
                    model_variable in ComponentSet(identify_variables(objective_function)))

                # Now, we add the relevant objective function(s) to the list of constraints containing model_variable
                constraints_with_given_variable.update(objective_functions_with_given_variable)

            # Create a list of all the edges that need to be created based on the constraints that contain the given
            # model_variable
            edges_between_nodes = list(combinations(constraints_with_given_variable, 2))

            # Update edge_weight_dict or edge_set based on the determined edges_between_nodes
            if weighted_graph:
                new_edge_weights = ComponentMap(
                    (edge, edge_weight_dict.get(edge, 0) + 1) for edge in edges_between_nodes)
                edge_weight_dict.update(new_edge_weights)
            else:
                new_edges = ComponentSet(edges_between_nodes)
                edge_set.update(new_edges)

    # Now, using edge_weight_dict or edge_set (based on whether the user wants a weighted or unweighted graph,
    # respectively), the networkX graph (model_graph) will be updated with all of the edges determined above
    if weighted_graph:
        for edge in edge_weight_dict:
            model_graph.add_edge(id(edge[0]), id(edge[1]), weight=edge_weight_dict[edge])
        edge_weight_dict.clear()
    else:
        for edge in edge_set:
            model_graph.add_edge(id(edge[0]), id(edge[1]))
        edge_set.clear()

    # Log especially significant information with the following logger function
    _event_log(model, model_graph)

    # If the user provided an argument for file_destination, an edge list and adjacency list will be
    # saved in a directory at this location
    if file_destination is not None:
        _write_to_file(model_graph, node_type=node_type, with_objective=with_objective,
                       weighted_graph=weighted_graph, file_destination=file_destination)

    # Return the networkX graph based on the given Pyomo optimization model

    mapping = nx.get_node_attributes(model_graph, 'node_name')
    nx.relabel_nodes(model_graph, mapping, copy=False)

    return model_graph


def _event_log(model, model_graph):
    """
    Logs information about the results of the code in _generate_model_graph

    This function takes in the same Pyomo model as _generate_community_graph and the model_graph generated by
    _generate_model_graph (which is a networkX graph of nodes and edges based on the Pyomo model). Then, some relevant
    information about the model and model_graph is found and then logged using the Python logger.
    This function is designed to be called by _generate_community_graph.

    Args:
        model (Block): a Pyomo model or block to be used for community detection
        model_graph: a networkX graph with nodes and edges based on the given Pyomo optimization model

    Returns:
        This function returns nothing; it is simply a function designed to log information helpful to the user.
    """

    number_of_variables = len(list(model.component_data_objects(Var, descend_into=True)))
    number_of_constraints = len(list(model.component_data_objects(Constraint, descend_into=True)))
    number_of_objectives = len(list(model.component_data_objects(Objective, descend_into=True)))
    number_of_nodes, number_of_edges = model_graph.number_of_nodes(), model_graph.number_of_edges()

    # logging.info
    logging.info("in _generate_model_graph: %s variables found in the model" % number_of_variables)
    logging.info("in _generate_model_graph: %s constraints found in the model" % number_of_constraints)
    logging.info("in _generate_model_graph: %s objective(s) found in the model" % number_of_objectives)
    logging.info("in _generate_model_graph: %s nodes found in the model" % number_of_nodes)
    logging.info("in _generate_model_graph: %s edges found in the model" % number_of_edges)

    # logging.warning
    if number_of_variables == 0:
        logging.warning("in _generate_model_graph: No variables found in the model")
    if number_of_constraints == 0:
        logging.warning("in _generate_model_graph: No constraints found in the model")
    if number_of_objectives == 0:
        logging.warning("in _generate_model_graph: No objective(s) found in the model")
    if number_of_nodes == 0:
        logging.warning("in _generate_model_graph: No nodes generated from the model")
    if number_of_edges == 0:
        logging.warning("in _generate_model_graph: No edges generated from the model")


def _write_to_file(model_graph, node_type, with_objective, weighted_graph, file_destination):
    """
    Saves an edge list and adjacency list in a new directory at a specified destination

    This function takes in model_graph, a networkX graph created from a Pyomo optimization model, and its relevant
    characteristics, and then uses the user-provided file_destination as a path to write an edge list and adjacency
    list for the given model_graph. If an invalid file path is given, this will be handled by making intermediate
    directories.
    This function is designed to be called by _generate_community_graph, which is in turn designed to be called by
    detect_communities

    Args:
        model_graph: a networkX graph created from a Pyomo optimization model
        node_type : a string that specifies the node_type of the graph; node_type='v' creates a graph with variable
        nodes and constraint edges. node_type='c' returns a graph with constraint nodes and variable edges, and any
        other input returns an error message
        with_objective: a Boolean argument that specifies whether or not the objective function will be
        treated as a node/constraint (depending on what node_type is specified as (see prior argument))
        weighted_graph: a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model
        file_destination: an optional argument that takes in a path if the user wants to save an edge and adjacency
        list based on the model

    Returns:
        This function returns nothing. Its only purpose is to write an edge list and adjacency list
        based on model_graph.
    """

    # Create a path based on the user-provided file_destination and the directory where the function will store the
    # edge list and adjacency list (community_detection_graph_info)
    community_detection_dir = os.path.join(file_destination, 'community_detection_graph_info')

    # In case the user-provided file_destination does not exist, create intermediate directories so that
    # community_detection_dir is now a valid path
    if not os.path.exists(community_detection_dir):
        os.makedirs(community_detection_dir)
        logging.error("in detect_communities: The given file path did not exist so the following file path was created"
                      "and used to store the edge list and adjacency list: %s" % community_detection_dir)

    # Collect information for naming the edge and adjacency lists:

    # Based on node_type, determine the type of node
    if node_type == 'v':
        type_of_node = 'variable'
    else:
        type_of_node = 'constraint'

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

    # Now, using all of this information, use the networkX functions to write the edge and adjacency lists to the
    # file path determined above and name them using the relevant graph information organized above
    nx.write_edgelist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                      '.%s_%s_edge_list_%s' % (type_of_node, weight_status, obj_status))
    nx.write_adjlist(model_graph, os.path.join(community_detection_dir, 'community_detection') +
                     '.%s_%s_adj_list_%s' % (type_of_node, weight_status, obj_status))
