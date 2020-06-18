"""
Main module for community detection integration with Pyomo models.

This module separates model variables or constraints into different communities
distinguished by the degree of connectivity between community members.

Original implementation developed by Rahul Joglekar in the Grossmann research group.

"""
from logging import getLogger

from pyomo.common.dependencies import attempt_import
from pyomo.core import ConcreteModel, ComponentMap
from pyomo.contrib.community_detection.community_graph import _generate_model_graph
from pyomo.common.dependencies import networkx as nx

import matplotlib.pyplot as plt
from matplotlib import cm

logger = getLogger('pyomo.contrib.community_detection')

# Attempt import of louvain community detection package
community_louvain, community_louvain_available = attempt_import(
    'community', error_message="Could not import the 'community' library, available via 'python-louvain' on PyPI.")


def detect_communities(model, node_type='c', with_objective=True, weighted_graph=True, random_seed=None):
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
                                         "values: 'b', 'c', 'v'" % node_type

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert random_seed is None or type(random_seed) == int, "Invalid value for random_seed: 'random_seed=%s' - " \
                                                            "random_seed must be a non-negative integer" % random_seed

    # Generate model_graph (a networkX graph based on the given Pyomo optimization model),
    # number_component_map (a dictionary to convert the communities into lists of Pyomo components
    # instead of node values), and constraint_variable_map (a dictionary that maps a constraint to the variables
    # it contains)
    model_graph, number_component_map, constraint_variable_map = _generate_model_graph(
        model, node_type=node_type, with_objective=with_objective, weighted_graph=weighted_graph)

    # Use Louvain community detection to find the communities
    # This returns a dictionary mapping individual nodes to their communities
    partition_of_graph = community_louvain.best_partition(model_graph, random_state=random_seed)

    # Now, use partition_of_graph to create a dictionary (community_map) that maps community numbers to their nodes
    number_of_communities = len(set(partition_of_graph.values()))
    community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)

    # At this point, we have community_map, which maps an integer (the community number) to a list of the node values
    # (which are just numbers) that correspond to the Pyomo modeling components in each community

    # Now, we want to include another list which will be determined based on the node_type given by the user, and
    # is described below
    # Thus, each key in community_map will map to a tuple of two lists, a constraint list and a variable list

    # Also, as this second list is constructed, the node values will be converted back to the Pyomo components
    # through the use of number_component_map, resulting in a dictionary where the values are two-list tuples that
    # contain Pyomo modeling components

    # Both variable and constraint nodes (bipartite graph) - for a given community, we simply want to separate the
    # nodes into their two groups; thus, we create a list of constraints and a list of variables
    if node_type == 'b':
        for community_key in community_map:
            constraint_node_list, variable_node_list = [], []
            node_community_list = community_map[community_key]
            for numbered_node in node_community_list:
                if numbered_node in constraint_variable_map:
                    constraint_node_list.append(number_component_map[numbered_node])
                else:
                    variable_node_list.append(number_component_map[numbered_node])
            community_map[community_key] = (constraint_node_list, variable_node_list)

    # Constraint node type - for a given community, we want to create a second list that contains all of the variables
    # contained in the constraints of that community
    elif node_type == 'c':
        for community_key in community_map:
            constraint_list = sorted(community_map[community_key])
            variable_list = [constraint_variable_map[numbered_constraint] for numbered_constraint in constraint_list]
            variable_list = sorted(set([node for variable_sublist in variable_list for node in variable_sublist]))
            variable_list = [number_component_map[variable] for variable in variable_list]
            constraint_list = [number_component_map[constraint] for constraint in constraint_list]
            community_map[community_key] = (constraint_list, variable_list)

    # Variable node type - for a given community, we want to create a second list that contains all of the constraints
    # that the variables of that community appear in
    elif node_type == 'v':
        for community_key in community_map:
            variable_list = sorted(community_map[community_key])
            constraint_list = []
            for numbered_variable in variable_list:
                constraint_list.extend([constraint_key for constraint_key in constraint_variable_map if
                                        numbered_variable in constraint_variable_map[constraint_key]])
            constraint_list = sorted(set(constraint_list))
            constraint_list = [number_component_map[constraint] for constraint in constraint_list]
            variable_list = [number_component_map[variable] for variable in variable_list]
            community_map[community_key] = (constraint_list, variable_list)

    # Log information about the number of communities found from the model
    logger.info("%s communities were found in the model" % number_of_communities)
    if number_of_communities == 0:
        logger.error("in detect_communities: Empty community map was returned")
    if number_of_communities == 1:
        logger.warning("Community detection found that with the given parameters, the model could not be decomposed - "
                       "only one community was found")

    # Return community_map, which has integer keys that now map to a tuple of two lists
    # containing Pyomo modeling components
    return community_map


def draw_model_graph(model, community_map=None, node_type='c', with_objective=True, weighted_graph=True,
                     random_seed=None, type_of_map='c'):
    """
    This function draws a graph of the communities for a Pyomo model.

    This function takes in a Pyomo model and its community map - if no community map is given, a community map is
    created with the detect_communities function (which uses Louvain community detection). A networkX graph of
    the model is created with the function _generate_model_graph(), using the parameters specified by the user.
    This model graph and the given parameters (node_type, with_objective, weighted_graph) are used to create
    the nodes and edges on the visualization of the model graph. The community map is used to color the nodes
    according to their communities, with preference being given to the first community a node is found
    in (this accounts for the possibility of a node being in more than one community).

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for community detection
    community_map: dict, optional
        a dictionary that maps an integer key (which corresponds to a community number) to a tuple of two lists.
        The first list is made up of the constraints in the given community and the second list is made up of the
        variables in the given community.
    node_type: str, optional
        A string that specifies the types of nodes to be drawn, the default is 'c'.
        'c' draws a graph with constraint nodes,
        'v' draws a graph with variable nodes,
        'b' draws a graph with both constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function will be
        included as a node/constraint (depending on what node_type is specified as (see prior argument)), the default
        is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether a weighted or unweighted graph is to be created from the Pyomo
        model; the default is True (node_type='b' creates an unweighted graph regardless of this parameter)
    random_seed: int, optional
        An integer that is used as the random seed for the heuristic Louvain community detection (only used if no community_map is given)
    type_of_map:
        This is used as the node_type in the function detect_communities to create a community_map, which will be used
        for coloring the graph and drawing the edges (only used if no community_map is given)

    Returns
    -------
    This function returns nothing; it is only meant to visualize the graph created from the Pyomo model
    """

    # Check that all arguments are of the correct type

    assert isinstance(model, ConcreteModel), "Invalid model: 'model=%s' - model must be an instance of " \
                                             "ConcreteModel" % model

    assert node_type in ('b', 'c', 'v'), "Invalid node type specified: 'node_type=%s' - Valid " \
                                         "values: 'b', 'c', 'v'" % node_type

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert random_seed is None or type(random_seed) == int, "Invalid value for random_seed: 'random_seed=%s' - " \
                                                            "random_seed must be a non-negative integer" % random_seed

    assert type_of_map in ('b', 'c', 'v'), "Invalid type of map specified: 'type_of_map=%s' - Valid " \
                                           "values: 'b', 'c', 'v'" % type_of_map

    # Use the _generate_model_graph function to create a networkX graph of the given model (along with
    # number_component_map and constraint_variable_map, which will be used to help with drawing the graph)
    model_graph, number_component_map, constraint_variable_map = _generate_model_graph(
        model, node_type=node_type, with_objective=with_objective, weighted_graph=weighted_graph)

    # This line creates the "reverse" of the number_component_map returned by _generate_model_graph above,
    # since component_number_map is more convenient to use in this function
    component_number_map = ComponentMap((comp, number) for number, comp in number_component_map.items())

    if community_map is None:
        # If no community map is given by the user, then a community map will be created using the given
        # model and the detect_communities function

        community_map = detect_communities(model, node_type=type_of_map, with_objective=with_objective,
                                           weighted_graph=weighted_graph, random_seed=random_seed)

    else:  # This is the case where the user has provided their own community_map

        # Check that the contents of the dictionary are of the right types
        assert type(community_map) == dict
        assert list(community_map.keys()) == [integer_key for integer_key in range(len(community_map))]
        for hopefully_a_tuple in list(community_map.values()):
            assert type(hopefully_a_tuple) == tuple
            assert type(hopefully_a_tuple[0]) == list
            assert type(hopefully_a_tuple[1]) == list
            assert len(hopefully_a_tuple) == 2

        for community_key in community_map:
            tuple_of_lists = community_map[community_key]
            for community_member in (tuple_of_lists[0] + tuple_of_lists[1]):
                assert community_member in component_number_map

    # Now we will use the component_number_map to change the Pyomo modeling components in community_map into the
    # numbers that correspond to their nodes/edges in the networkX graph, model_graph
    for key in community_map:
        community_map[key] = ([component_number_map[component] for component in community_map[key][0]],
                              [component_number_map[component] for component in community_map[key][1]])

    # Based on node_type, which specifies what Pyomo modeling components are to be drawn as nodes in the graph
    # drawing, we will now get the node list and the color list, which describes how to color nodes
    # according to their communities (which is based on community_map)
    if node_type == 'b':
        nonflattened_list_of_nodes = [list_of_nodes for list_tuple in community_map.values() for list_of_nodes in
                                      list_tuple]

        # nonflattened_list_of_nodes is a list of lists, so we will use the one-list comprehension below to flatten
        # the list and get our one-dimensional node list
        node_list = [node for sublist in nonflattened_list_of_nodes for node in sublist]

        color_list = []
        # Now, we will find the first community that a node appears in and color the node based on that community
        # (because in community_map, certain nodes may appear in multiple communities)
        for node in node_list:
            not_found = True
            for community_key in community_map:
                if not_found and node in (community_map[community_key][0] + community_map[community_key][1]):
                    color_list.append(community_key)
                    not_found = False

        # Find top_nodes (one of the two "groups" of nodes in a bipartite graph) for
        # the pos argument in the nx.draw_networkx_nodes function
        if model_graph.number_of_nodes() > 0 and nx.is_connected(model_graph):
            top_nodes = nx.bipartite.sets(model_graph)[1]
        else:
            top_nodes = {node for node in model_graph.nodes() if node in constraint_variable_map}
        pos = nx.bipartite_layout(model_graph, top_nodes)

    else:
        # This is in the case that node_type is 'c' or 'v'

        # Constraints should be in the first list in the tuple and variables should be in the second list
        position = 0 if node_type == 'c' else 1
        nonflattened_list_of_nodes = list(i[position] for i in community_map.values())

        # nonflattened_list_of_nodes is a list of lists, so we will use the one-list comprehension below to flatten
        # the list and get our one-dimensional node list
        node_list = [node for sublist in nonflattened_list_of_nodes for node in sublist]

        # Now, we will find the first community that a node appears in and color the node based on that community
        # (because in community_map, certain nodes may appear in multiple communities)
        color_list = []
        for node in node_list:
            not_found = True
            for community_key in community_map:
                if not_found and node in community_map[community_key][position]:
                    color_list.append(community_key)
                    not_found = False

        # Note - there is no strong reason to choose spring layout; it just creates relatively clean graphs
        pos = nx.spring_layout(model_graph)

    # Define color_map
    color_map = cm.get_cmap('viridis', len(community_map))
    # Draw the graph
    nx.draw_networkx_nodes(model_graph, pos, nodelist=node_list, node_size=40, cmap=color_map, node_color=color_list)
    nx.draw_networkx_edges(model_graph, pos, alpha=0.5)
    # Display the graph
    plt.show()


def easy_to_read(model=None, community_map=None, node_type='c', with_objective=True, weighted_graph=True,
                 random_seed=None):
    """
    This function takes in a community map of Pyomo components and returns the same community map with the strings
    of the Pyomo components or takes in a model and returns a community map (based on Louvain community detection)
    with the strings of Pyomo components

    Parameters
    ----------
    model: Block, optional
        a Pyomo model or block to be used for community detection
    community_map: dict, optional
        a dictionary with values that contain Pyomo components which will be converted to their strings
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

    Returns
    -------
    community_map: dict
        a Python dictionary whose keys are integers from zero to the number of communities minus one
        with values that are sorted lists of the nodes in the given community
    """

    # Check that arguments are of the right type

    assert model is not None or community_map is not None, "Either a model or a community map must be given as an input"

    assert isinstance(model, ConcreteModel) or model is None, "Invalid model: 'model=%s' - model must be " \
                                                              "an instance of ConcreteModel or None " \
                                                              "if a community map is given" % model

    assert node_type in ('b', 'c', 'v'), "Invalid node type specified: 'node_type=%s' - Valid " \
                                         "values: 'b', 'c', 'v'" % node_type

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert random_seed is None or type(random_seed) == int, "Invalid value for random_seed: 'random_seed=%s' - " \
                                                            "random_seed must be a non-negative integer" % random_seed

    if community_map is None:
        # If no community map is given by the user, then a community map of strings will be created using the given
        # model and the easy_to_read function
        community_map = detect_communities(model, node_type=node_type, with_objective=with_objective,
                                           weighted_graph=weighted_graph, random_seed=random_seed)
    else:
        # Make sure community_map is of the correct format
        assert type(community_map) == dict
        assert list(community_map.keys()) == [integer_key for integer_key in range(len(community_map))]
        for hopefully_a_tuple in list(community_map.values()):
            assert type(hopefully_a_tuple) == tuple
            assert type(hopefully_a_tuple[0]) == list
            assert type(hopefully_a_tuple[1]) == list
            assert len(hopefully_a_tuple) == 2

        # TODO: Add assert statement for list items (to check that they are Pyomo modeling components)

        # As of right now, there is no check to make sure all of the items in the lists are Pyomo modeling
        # components; currently waiting on an elegant way to check this

        # Current best idea is to have something like this for every list item in the community map:
        # assert isinstance(list_item,
        # (Var, Constraint, Objective, _GeneralVarData, _GeneralConstraintData, _GeneralObjectiveData))

        # Only having Var, Constraint, Objective doesn't seem to catch indexed components or piecewise components

    # Convert the components in community_map to their strings
    for key in community_map:
        community_map[key] = ([str(component) for component in community_map[key][0]],
                              [str(component) for component in community_map[key][1]])

    return community_map
