"""Model Graph Generator Code"""
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.common.dependencies import networkx as nx
from pyomo.core import Constraint, Objective, Var
from pyomo.core.expr.current import identify_variables
from itertools import combinations
import logging


def _generate_model_graph(model, node_type, with_objective, weighted_graph):
    """
    Creates a networkX graph of nodes and edges based on a Pyomo optimization model

    This function takes in a Pyomo optimization model, then creates a graphical representation of the model with
    specific features of the graph determined by the user (see Parameters below).

    As for how the graphs are created:
    If the user chooses constraint nodes, then the edge between two given nodes is created if those two constraint
    equations share a common variable. The weight of each edge depends on the number of variables common to the two
    constraint equations.
    If the user chooses variable nodes, then the edge between two given nodes is created if those two variables occur
    together in the same constraint equation. The weight of each edge depends on the number of constraint equations
    in which the two variables occur together.
    If the user chooses both constraint and variable nodes (bipartite graph), then edges will only be created between a
    variable and a constraint (not between two nodes of the same type). The weight of each edge depends on the number
    of times a variable appears in a constraint.

    This function is designed to be called by detect_communities, get_edge_list, or get_adj_list.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for community detection
    node_type: str
        A string that specifies the type of graph that is created from the model
        'c' creates a graph based on constraint nodes,
        'v' creates a graph based on variable nodes,
        'b' creates a graph based on constraint and variable nodes (bipartite graph).
    with_objective: bool
        a Boolean argument that specifies whether or not the objective function will be
        included as a node/constraint (depending on what node_type is specified as (see prior argument))
    weighted_graph: bool
        a Boolean argument that specifies whether a weighted or unweighted graph is to be
        created from the Pyomo model

    Returns
    -------
    model_graph: networkX graph
        a networkX graph with nodes and edges based on the given Pyomo optimization model
    string_map: dict
        a dictionary that maps the string of a Pyomo modeling component (such as a variable) to the actual component;
        will be used to convert a string back to the actual modeling component
    constraint_variable_map: dict
        a dictionary that maps the string of a constraint to a list of the strings of the variables in the constraint
    """

    # Initialize the data structure needed to keep track of edges in the graph
    edge_set = set()

    # Initialize the three items to be returned by this function
    model_graph = nx.Graph()
    collapsed_model_graph = nx.Graph()
    collapsed_model_graph_nodes = []
    constraint_variable_map = {}  # this will not be used if node_type = 'b'
    string_map = {}

    # Initialize set to check membership for type of node
    variable_node_set = set()

    # Create nodes based on variables in the Pyomo model
    for model_variable in model.component_data_objects(Var, descend_into=True):
        model_graph.add_node(str(model_variable))

        # Update string_map
        string_map[str(model_variable)] = model_variable

        # Update variable_node_set
        variable_node_set.add(str(model_variable))

    # Loop through all constraints in the Pyomo model in the Pyomo model to determine what edges need to be created
    for model_constraint in model.component_data_objects(Constraint, descend_into=True):
        # Create nodes based on constraints in the Pyomo model
        model_graph.add_node(str(model_constraint))

        # Create a ComponentSet of the variables that occur in the given constraint equation
        variables_in_constraint_equation = ComponentSet(identify_variables(model_constraint.body))

        # Update constraint_variable_map and string_map
        if node_type in ('c', 'v'):
            constraint_variable_map[str(model_constraint)] = [str(variable) for variable in
                                                              variables_in_constraint_equation]
        string_map[str(model_constraint)] = model_constraint

        # Create a list of all the edges that need to be created based on this constraint equation
        edges_between_nodes = [(model_constraint, variable_in_constraint)
                               for variable_in_constraint in variables_in_constraint_equation]

        # Update edge_set based on the determined edges_between_nodes
        new_edges = {tuple(sorted([str(edge[0]), str(edge[1])])) for edge in edges_between_nodes}
        edge_set.update(new_edges)

    # This if statement will be executed if the user chooses to include the objective function as a node in
    # this model graph
    if with_objective:
        # Use a loop to account for the possibility of multiple objective functions
        for objective_function in model.component_data_objects(Objective, descend_into=True):
            # Add objective_function as a node in model_graph
            model_graph.add_node(str(objective_function))

            # Create a ComponentSet of the variables that occur in the given objective function
            variables_in_objective_function = ComponentSet(identify_variables(objective_function))

            # Update constraint_variable_map and string_map
            if node_type in ('c', 'v'):
                constraint_variable_map[str(objective_function)] = [str(variable) for variable in
                                                                    variables_in_objective_function]
            string_map[str(objective_function)] = objective_function

            # Create a list of all the edges that need to be created based on the variables in the objective
            edges_between_nodes = [(objective_function, variable_in_constraint)
                                   for variable_in_constraint in variables_in_objective_function]

            # Update edge_set based on the determined edges_between_nodes
            new_edges = {tuple(sorted([str(edge[0]), str(edge[1])])) for edge in edges_between_nodes}
            edge_set.update(new_edges)

    model_graph_edges = sorted(edge_set)
    model_graph.add_edges_from(model_graph_edges)
    del edge_set

    # Constraint nodes
    if node_type == 'c':
        if weighted_graph:
            edge_weight_dict = dict()
        else:
            edge_set = set()

        for node in model_graph.nodes():
            if node in variable_node_set:
                connected_constraints = []

                for edge in model_graph.edges(node):
                    connected_constraints.append(edge[1])
                edges_between_constraints = list(combinations(connected_constraints, 2))

                # Update edge_weight_dict or edge_set based on the determined edges_between_nodes
                if weighted_graph:
                    for edge in edges_between_constraints:
                        new_edge = tuple(sorted([str(edge[0]), str(edge[1])]))
                        new_edge_weights = {new_edge: edge_weight_dict.get(new_edge, 0) + 1}
                        edge_weight_dict.update(new_edge_weights)
                else:
                    new_edges = {tuple(sorted([str(edge[0]), str(edge[1])])) for edge in edges_between_constraints}
                    edge_set.update(new_edges)
            else:
                collapsed_model_graph_nodes.append(node)

    # Variable nodes
    elif node_type == 'v':
        if weighted_graph:
            edge_weight_dict = dict()
        else:
            edge_set = set()

        for node in model_graph.nodes():
            if node not in variable_node_set:
                connected_variables = []

                for edge in model_graph.edges(node):
                    connected_variables.append(edge[1])
                edges_between_variables = list(combinations(connected_variables, 2))

                # Update edge_weight_dict or edge_set based on the determined edges_between_nodes
                if weighted_graph:
                    for edge in edges_between_variables:
                        new_edge = tuple(sorted([str(edge[0]), str(edge[1])]))
                        new_edge_weights = {new_edge: edge_weight_dict.get(new_edge, 0) + 1}
                        edge_weight_dict.update(new_edge_weights)
                else:
                    new_edges = {tuple(sorted([str(edge[0]), str(edge[1])])) for edge in edges_between_variables}
                    edge_set.update(new_edges)
            else:
                collapsed_model_graph_nodes.append(node)

    # Bipartite graph - both variable and constraint nodes
    elif node_type == 'b':
        return model_graph, string_map, constraint_variable_map

    collapsed_model_graph.add_nodes_from(sorted(collapsed_model_graph_nodes))

    # Now, using edge_weight_dict or edge_set (based on if the user wants a weighted graph or an unweighted graph,
    # respectively), the networkX graph (collapsed_model_graph) will be updated with all of the edges determined above
    if weighted_graph:
        collapsed_model_graph_edges = sorted(edge_weight_dict)
        collapsed_model_graph.add_edges_from(collapsed_model_graph_edges)

        seen_edges = set()
        for edge in edge_weight_dict:
            node_one = edge[0]
            node_two = edge[1]
            if edge in seen_edges:
                collapsed_model_graph[node_one][node_two]['weight'] += edge_weight_dict[edge]
            else:
                collapsed_model_graph[node_one][node_two]['weight'] = edge_weight_dict[edge]
                seen_edges.add(edge)

        del edge_weight_dict
        del seen_edges

    else:
        collapsed_model_graph_edges = sorted(edge_set)
        collapsed_model_graph.add_edges_from(collapsed_model_graph_edges)
        del edge_set

    # Log important information with the following logger function
    _event_log(model, model_graph)

    # Return the networkX graph based on the given Pyomo optimization model, string_map, and constraint_variable_map
    return collapsed_model_graph, string_map, constraint_variable_map


def _event_log(model, model_graph):
    """
    Logs information about the results of the code in _generate_model_graph

    This function takes in the same Pyomo model as _generate_community_graph and the model_graph generated by
    _generate_model_graph (which is a networkX graph of nodes and edges based on the Pyomo model). Then, some relevant
    information about the model and model_graph is determined and logged using the logger.
    This function is designed to be called by _generate_community_graph.

    Args:
        model (Block): a Pyomo model or block to be used for community detection
        model_graph: a networkX graph with nodes and edges based on the given Pyomo optimization model

    Returns:
        This function returns nothing; it simply logs information that is relevant to the user.
    """

    # Collect some information that will be useful for the logger
    number_of_variables = len(list(model.component_data_objects(Var, descend_into=True)))
    number_of_constraints = len(list(model.component_data_objects(Constraint, descend_into=True)))
    number_of_objectives = len(list(model.component_data_objects(Objective, descend_into=True)))
    number_of_nodes, number_of_edges = model_graph.number_of_nodes(), model_graph.number_of_edges()

    # Log this information as info
    logging.info("in _generate_model_graph: %s variables found in the model" % number_of_variables)
    logging.info("in _generate_model_graph: %s constraints found in the model" % number_of_constraints)
    logging.info("in _generate_model_graph: %s objective(s) found in the model" % number_of_objectives)
    logging.info("in _generate_model_graph: %s nodes found in the model" % number_of_nodes)
    logging.info("in _generate_model_graph: %s edges found in the model" % number_of_edges)

    # Given one of the conditionals below, we will log this information as a warning
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
