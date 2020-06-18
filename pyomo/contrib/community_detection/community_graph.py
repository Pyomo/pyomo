"""Model Graph Generator Code"""
from logging import getLogger

from itertools import combinations
from pyomo.common.dependencies import networkx as nx
from pyomo.core import Constraint, Objective, Var, ComponentMap, SortComponents
from pyomo.core.expr.current import identify_variables

logger = getLogger('pyomo.contrib.community_detection')


def _generate_model_graph(model, node_type, with_objective, weighted_graph):
    """
    Creates a networkX graph of nodes and edges based on a Pyomo optimization model

    This function takes in a Pyomo optimization model, then creates a graphical representation of the model with
    specific features of the graph determined by the user (see Parameters below).

    This function is designed to be called by detect_communities or visualize_model_graph.

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
    bipartite_model_graph/collapsed_model_graph: networkX graph
        a networkX graph with nodes and edges based on the given Pyomo optimization model
    number_component_map: dict
        a dictionary that has a deterministic mapping of a number to a Pyomo modeling component
    constraint_variable_map: dict
        a dictionary that maps a numbered constraint to a list of (numbered) variables that appear in the constraint
        (the numbers' mapping to the Pyomo components is given in number_component_map)
    """

    # Start off by making a bipartite graph (regardless of node_type), then if node_type = 'v' or 'c',
    # "collapse" this bipartite graph into a variable node or constraint node graph

    # Initialize the data structure needed to keep track of edges in the graph (this graph will be made
    # without edge weights, because edge weights are not useful for this bipartite graph)
    edge_set = set()

    bipartite_model_graph = nx.Graph()  # Initialize networkX graph for the bipartite graph
    constraint_variable_map = {}  # Initialize map of the variables in constraint equations

    # Make a dict of all the components we need for the networkX graph (since we cannot use the components directly
    # in the networkX graph)
    if with_objective:
        component_number_map = ComponentMap((component, number) for number, component in enumerate(
            model.component_data_objects(ctype=(Constraint, Var, Objective), active=True, descend_into=True,
                                         sort=SortComponents.deterministic)))
    else:
        component_number_map = ComponentMap((component, number) for number, component in enumerate(
            model.component_data_objects(ctype=(Constraint, Var), active=True, descend_into=True,
                                         sort=SortComponents.deterministic)))

    # Create the reverse of component_number_map, which will be used in detect_communities to convert the node numbers
    # to their corresponding Pyomo modeling components
    number_component_map = dict((number, comp) for comp, number in component_number_map.items())

    # Add the components as nodes to the bipartite graph
    bipartite_model_graph.add_nodes_from([node_number for node_number in range(len(component_number_map))])

    # Loop through all constraints in the Pyomo model to determine what edges need to be created
    for model_constraint in model.component_data_objects(ctype=Constraint, active=True, descend_into=True):
        numbered_constraint = component_number_map[model_constraint]

        # Create a list of the variable numbers that occur in the given constraint equation
        numbered_variables_in_constraint_equation = [component_number_map[constraint_variable]
                                                     for constraint_variable in
                                                     identify_variables(model_constraint.body)]

        # Update constraint_variable_map
        constraint_variable_map[numbered_constraint] = numbered_variables_in_constraint_equation

        # Create a list of all the edges that need to be created based on the variables in this constraint equation
        edges_between_nodes = [(numbered_constraint, numbered_variable_in_constraint)
                               for numbered_variable_in_constraint in numbered_variables_in_constraint_equation]

        # Update edge_set based on the determined edges between nodes
        edge_set.update(edges_between_nodes)

    # This if statement will be executed if the user chooses to include the objective function as a node in
    # the model graph
    if with_objective:

        # Use a loop to account for the possibility of multiple objective functions
        for objective_function in model.component_data_objects(ctype=Objective, active=True, descend_into=True):
            numbered_objective = component_number_map[objective_function]

            # Create a list of the variable numbers that occur in the given objective function
            numbered_variables_in_objective = [component_number_map[objective_variable]
                                               for objective_variable in identify_variables(objective_function)]

            # Update constraint_variable_map
            constraint_variable_map[numbered_objective] = numbered_variables_in_objective

            # Create a list of all the edges that need to be created based on the variables in the objective function
            edges_between_nodes = [(numbered_objective, numbered_variable_in_objective)
                                   for numbered_variable_in_objective in numbered_variables_in_objective]

            # Update edge_set based on the determined edges between nodes
            edge_set.update(edges_between_nodes)

    bipartite_model_graph.add_edges_from(sorted(edge_set))

    # Both variable and constraint nodes (bipartite graph); this is exactly the graph we made above
    if node_type == 'b':
        # Log important information with the following logger function
        _event_log(model, bipartite_model_graph, set(constraint_variable_map), node_type, with_objective)

        # Return the bipartite networkX graph, the dictionary of node numbers mapped to their respective Pyomo
        # components, and the map of constraints to the variables they contain
        return bipartite_model_graph, number_component_map, constraint_variable_map

    # If we reach this point of the code, then we will now begin constructing the collapsed version of the bipartite
    # model graph (the specific manner depends on whether node type is 'c' or 'v')
    if weighted_graph:
        edge_weight_dict = dict()
    else:
        edge_set = set()

    collapsed_model_graph = nx.Graph()  # Initialize networkX graph for the collapsed version of bipartite_model_graph

    # Constraint nodes - now we will collapse the bipartite graph into a constraint node graph
    if node_type == 'c':

        for node in bipartite_model_graph.nodes():

            # If the node represents a variable
            if node not in constraint_variable_map:
                connected_constraints = []  # Initialize list of constraints that share this variable

                # Loop through the variable node's edges to find constraints that contain the variable
                for edge in bipartite_model_graph.edges(node):
                    # The first node in the edge tuple will always be the node that is used as the
                    # argument in 'bipartite_model_graph.edges(node)'
                    # Thus, the relevant node is the second one in the tuple 'edge[1]'
                    connected_constraints.append(edge[1])

                # Create all possible two-node combinations from connected_constraints; in other words, create a list
                # of all the edges that need to be created between constraints based on connected_constraints
                edges_between_constraints = list(combinations(sorted(connected_constraints), 2))

                # Update edge_weight_dict or edge_set based on the determined edges between nodes
                if weighted_graph:
                    new_edge_weights = {edge: edge_weight_dict.get(edge, 0) + 1 for edge in edges_between_constraints}
                    edge_weight_dict.update(new_edge_weights)
                else:
                    edge_set.update(edges_between_constraints)

            # If the node is not a variable, then it is a constraint and we want it as
            # a node in our constraint node graph
            else:
                collapsed_model_graph.add_node(node)

    # Variable nodes - now we will collapse the bipartite graph into a variable node graph
    elif node_type == 'v':

        for node in bipartite_model_graph.nodes():

            # If the node represents a constraint (or objective)
            if node in constraint_variable_map:

                # Create list of variables in the constraint equation
                connected_variables = constraint_variable_map[node]

                # Create all possible two-node combinations from connected_variables; in other words, create a list
                # of all the edges that need to be created between variables based on connected_variables
                edges_between_variables = list(combinations(sorted(connected_variables), 2))

                # Update edge_weight_dict or edge_set based on the determined edges_between_nodes
                if weighted_graph:
                    new_edge_weights = {edge: edge_weight_dict.get(edge, 0) + 1 for edge in edges_between_variables}
                    edge_weight_dict.update(new_edge_weights)
                else:
                    edge_set.update(edges_between_variables)

            # If the node is not a constraint/objective, then it is a variable and we want it as
            # a node in our variable node graph
            else:
                collapsed_model_graph.add_node(node)

    # Now, using edge_weight_dict or edge_set (based on if the user wants a weighted graph or an unweighted graph,
    # respectively), the networkX graph (collapsed_model_graph) will be updated with all of the edges determined above
    if weighted_graph:

        # Add edges to collapsed_model_graph
        collapsed_model_graph.add_edges_from(sorted(edge_weight_dict))

        # Iterate through the edges in edge_weight_dict and add them to collapsed_model_graph
        for edge in edge_weight_dict:
            node_one, node_two = edge[0], edge[1]
            collapsed_model_graph[node_one][node_two]['weight'] = edge_weight_dict[edge]

    else:
        # Add edges to collapsed_model_graph
        collapsed_model_graph.add_edges_from(sorted(edge_set))

    # Log important information with the following logger function
    _event_log(model, collapsed_model_graph, set(constraint_variable_map), node_type, with_objective)

    # Return the collapsed networkX graph, the dictionary of node numbers mapped to their respective Pyomo
    # components, and the map of constraints to the variables they contain
    return collapsed_model_graph, number_component_map, constraint_variable_map


def _event_log(model, model_graph, constraint_set, node_type, with_objective):
    """
    Logs information about the results of the code in _generate_model_graph

    This function takes in the same Pyomo model as _generate_community_graph and the model_graph generated by
    _generate_model_graph (which is a networkX graph of nodes and edges based on the Pyomo model). Then, some relevant
    information about the model and model_graph is determined and logged using the logger.

    This function is designed to be called by _generate_community_graph.

    Parameters
    ----------
    model: Block
         the Pyomo model or block to be used for community detection
    model_graph: networkX graph
        a networkX graph with nodes and edges based on the given Pyomo optimization model

    Returns
    -------
    This function returns nothing; it simply logs information that is relevant to the user.
    """

    # Collect some information that will be useful for the logger
    all_variables_count = len(list(model.component_data_objects(ctype=Var, descend_into=True)))

    active_constraints_count = len(list(model.component_data_objects(ctype=Constraint, active=True, descend_into=True)))
    all_constraints_count = len(list(model.component_data_objects(ctype=Constraint, descend_into=True)))

    active_objectives_count = len(list(model.component_data_objects(ctype=Objective, active=True, descend_into=True)))
    all_objectives_count = len(list(model.component_data_objects(ctype=Objective, descend_into=True)))

    number_of_nodes, number_of_edges = model_graph.number_of_nodes(), model_graph.number_of_edges()

    # Log this information as info
    logger.info("%s variables found in the model" % all_variables_count)

    logger.info("%s constraints found in the model" % all_constraints_count)
    logger.info("%s active constraints found in the model" % active_constraints_count)

    logger.info("%s objective(s) found in the model" % all_objectives_count)
    logger.info("%s active objective(s) found in the model" % active_objectives_count)

    logger.info("%s nodes found in the graph created from the model" % number_of_nodes)
    logger.info("%s edges found in the graph created from the model" % number_of_edges)

    # Log information on connectivity and density
    if number_of_nodes > 0:
        if nx.is_connected(model_graph):
            logger.info("The graph created from the model is connected.")
            graph_is_connected = True
        else:
            logger.info("The graph created from the model is disconnected.")
            graph_is_connected = False

        if node_type == 'b':
            if graph_is_connected:
                top_nodes, bottom_nodes = nx.bipartite.sets(model_graph)
                if len(top_nodes) == 0:
                    top_nodes, bottom_nodes = bottom_nodes, top_nodes
                if list(top_nodes)[0] in constraint_set:
                    constraint_nodes = top_nodes
                    variable_nodes = bottom_nodes
                else:
                    constraint_nodes = bottom_nodes
                    variable_nodes = top_nodes
            else:
                constraint_nodes = {node for node in model_graph.nodes() if node in constraint_set}
                variable_nodes = set(model_graph) - constraint_nodes

            constraint_density = round(nx.bipartite.density(model_graph, constraint_nodes), 2)
            variable_density = round(nx.bipartite.density(model_graph, variable_nodes), 2)

            if constraint_density == 1 or variable_density == 1:  # If the graph is complete, both will equal 1
                logger.warning("The bipartite graph constructed from the model is complete (graph density equals 1)")
            else:
                logger.info(
                    "For the bipartite graph constructed from the model, the density for constraint nodes is %s" %
                    constraint_density)
                logger.info(
                    "For the bipartite graph constructed from the model, the density for variable nodes is %s" %
                    variable_density)

        else:
            graph_density = round(nx.density(model_graph), 2)

            if graph_density == 1:
                logger.warning("The graph constructed from the model is complete (graph density equals 1)")
            else:
                logger.info("The graph constructed from the model has a density of %s" % graph_density)

    # Given one of the conditionals below, we will log this information as a warning
    if all_variables_count == 0:
        logger.warning("No variables found in the model")

    if all_constraints_count == 0:
        logger.warning("No constraints found in the model")
    elif active_constraints_count == 0:
        logger.warning("No active constraints found in the model")

    if all_objectives_count == 0:
        if with_objective:
            logger.warning("No objective(s) found in the model")
        else:
            logger.info("No objective(s) found in the model")
    elif active_objectives_count == 0:
        if with_objective:
            logger.warning("No active objective(s) found in the model")
        else:
            logger.info("No active objective(s) found in the model")

    if number_of_nodes == 0:
        logger.warning("No nodes were created for the graph (based on the model and the given parameters)")
    if number_of_edges == 0:
        logger.warning("No edges were created for the graph (based on the model and the given parameters)")
