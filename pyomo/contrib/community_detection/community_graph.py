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
        a Boolean argument that specifies whether a weighted or unweighted graph is to be created from the Pyomo
        model; the default is True (node_type='b' creates an unweighted graph regardless of this parameter)

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

    # Start off by making a bipartite graph (regardless of node_type), then if node_type = 'v' or 'c',
    # "collapse" this bipartite graph into a variable node or constraint node graph

    # Initialize the data structure needed to keep track of edges in the graph (this graph will be made
    # without edge weights, because edge weights are not useful for this bipartite graph)
    edge_set = set()

    model_graph = nx.Graph()  # Initialize networkX graph for the bipartite graph
    collapsed_model_graph_nodes = []  # Initialize list of nodes to be created for the collapsed graph
    constraint_variable_map = {}  # Initialize map of variables in constraints; this will not be used if node_type = 'b'
    string_map = {}  # Initialize dictionary to map between strings of modeling components and the actual components
    variable_node_set = set()  # Initialize set to check membership for type of node

    # Loop through all variables in the Pyomo model
    for model_variable in model.component_data_objects(Var, descend_into=True):
        # Create nodes based on variables in the Pyomo model
        model_graph.add_node(str(model_variable))

        # Update string_map
        string_map[str(model_variable)] = model_variable

        # Update variable_node_set
        variable_node_set.add(str(model_variable))

    # Loop through all constraints in the Pyomo model to determine what edges need to be created
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
    # the model graph
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

            # Create a list of all the edges that need to be created based on the variables in the objective function
            edges_between_nodes = [(objective_function, variable_in_constraint)
                                   for variable_in_constraint in variables_in_objective_function]

            # Update edge_set based on the determined edges_between_nodes
            new_edges = {tuple(sorted([str(edge[0]), str(edge[1])])) for edge in edges_between_nodes}
            edge_set.update(new_edges)

    model_graph_edges = sorted(edge_set)  # Create list from edges in edge_set
    model_graph.add_edges_from(model_graph_edges)
    del edge_set

    # Both variable and constraint nodes (bipartite graph); this is the graph we made above
    if node_type == 'b':
        return model_graph, string_map, variable_node_set

    # Constraint nodes
    elif node_type == 'c':
        if weighted_graph:
            edge_weight_dict = dict()
        else:
            edge_set = set()

        for node in model_graph.nodes():
            if node in variable_node_set:  # If the node represents a variable
                connected_constraints = []  # Initialize list of constraints that share this variable

                # Loop through the variable node's edges
                for edge in model_graph.edges(node):
                    # The first node in the edge tuple will always be the node fed into 'model_graph.edges(node)';
                    # thus, the relevant node is the second one in the tuple 'edge[1]'
                    connected_constraints.append(edge[1])

                # Create all possible two-node combinations from connected_constraints; in other words, create a list
                # of all the edges that need to be created between constraints based on connected_constraints
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

            # If the node is not a variable, then we want it as a node in our constraint node graph
            else:
                collapsed_model_graph_nodes.append(node)

    # Variable nodes
    elif node_type == 'v':
        if weighted_graph:
            edge_weight_dict = dict()
        else:
            edge_set = set()

        for node in model_graph.nodes():
            if node not in variable_node_set:  # If the node represents a constraint (or objective)

                # Create list of variables in the constraint equation
                connected_variables = constraint_variable_map[node]

                # Create all possible two-node combinations from connected_variables; in other words, create a list
                # of all the edges that need to be created between variables based on connected_variables
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

            # If the node is not a constraint/objective, then we want it as a node in our variable node graph
            else:
                collapsed_model_graph_nodes.append(node)

    collapsed_model_graph = nx.Graph()  # Initialize networkX graph for the collapsed version of model_graph
    collapsed_model_graph.add_nodes_from(sorted(collapsed_model_graph_nodes))

    # Now, using edge_weight_dict or edge_set (based on if the user wants a weighted graph or an unweighted graph,
    # respectively), the networkX graph (collapsed_model_graph) will be updated with all of the edges determined above
    if weighted_graph:

        # Add edges to collapsed_model_graph
        collapsed_model_graph_edges = sorted(edge_weight_dict)
        collapsed_model_graph.add_edges_from(collapsed_model_graph_edges)

        # Iterate through the edges in edge_weight_dict and add them to collapsed_model_graph
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
        # Add edges to collapsed_model_graph
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
