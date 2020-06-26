"""
Main module for community detection integration with Pyomo models.

This module separates model components (variables, constraints, and objectives) into different communities
distinguished by the degree of connectivity between community members.

Original implementation developed by Rahul Joglekar in the Grossmann research group.

"""
from logging import getLogger

from pyomo.common.dependencies import attempt_import
from pyomo.core import ConcreteModel, ComponentMap
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.common.dependencies import networkx as nx

logger = getLogger('pyomo.contrib.community_detection')

# Attempt import of louvain community detection package
community_louvain, community_louvain_available = attempt_import(
    'community', error_message="Could not import the 'community' library, available via 'python-louvain' on PyPI.")

# Attempt import of matplotlib
matplotlib, matplotlib_available = attempt_import('matplotlib', error_message="Could not import 'matplotlib'")

if matplotlib_available:
    import matplotlib.pyplot as plt
    from matplotlib import cm

# TODO: Consider adding an option to include inactive constraints/objectives in the community detection

def detect_communities(model, type_of_community_map='constraint', with_objective=True, weighted_graph=True,
                       random_seed=None):
    """
    Detects communities in a Pyomo optimization model

    This function takes in a Pyomo optimization model and organizes the variables and constraints into a graph of nodes
    and edges. Then, by using Louvain community detection on the graph, a dictionary (community_map) is created, which
    maps (arbitrary) community keys to the detected communities within the model.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for community detection
    type_of_community_map: str, optional
        a string that specifies the type of community map to be returned, the default is 'constraint'.
        'constraint' returns a dictionary (community_map) with communities based on constraint nodes,
        'variable' returns a dictionary (community_map) with communities based on variable nodes,
        'bipartite' returns a dictionary (community_map) with communities based on constraint and variable nodes (bipartite
        graph)
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function is
        included in the model graph (and thus in the community map); the default is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether the community map is created based on a weighted model graph or an
        unweighted model graph; the default is True (type_of_community_map='bipartite' creates an unweighted model graph
        regardless of this parameter)
    random_seed: int, optional
        an integer that is used as the random seed for the (heuristic) Louvain community detection

    Returns
    -------
    community_map: dict
        a Python dictionary that maps arbitrary keys (in this case, integers from zero to the number of
        communities minus one) to two-list tuples containing Pyomo components in the given community
    """

    # Check that all arguments are of the correct type
    assert isinstance(model, ConcreteModel), "Invalid model: 'model=%s' - model must be an instance of " \
                                             "ConcreteModel" % model

    assert type_of_community_map in ('bipartite', 'constraint', 'variable'), "Invalid value for type_of_community_map: " \
                                                                     "'type_of_community_map=%s' - Valid values: 'bipartite', 'constraint', 'variable'" \
                                                                     % type_of_community_map

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert random_seed is None or (type(random_seed) == int and random_seed >= 0), \
        "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed

    # Generate model_graph (a NetworkX graph based on the given Pyomo optimization model),
    # number_component_map (a dictionary to convert the communities into lists of Pyomo components
    # instead of number values), and constraint_variable_map (a dictionary that maps a constraint to the variables
    # it contains)
    model_graph, number_component_map, constraint_variable_map = generate_model_graph(
        model, type_of_graph=type_of_community_map, with_objective=with_objective, weighted_graph=weighted_graph)

    # Use Louvain community detection to find the communities - this returns a dictionary mapping
    # individual nodes to their communities
    partition_of_graph = community_louvain.best_partition(model_graph, random_state=random_seed)

    # Now, use partition_of_graph to create a dictionary (community_map) that maps community keys to the nodes
    # in each community
    number_of_communities = len(set(partition_of_graph.values()))
    community_map = {nth_community: [] for nth_community in range(number_of_communities)}
    for node in partition_of_graph:
        nth_community = partition_of_graph[node]
        community_map[nth_community].append(node)

    # At this point, we have community_map, which maps an integer (the community number) to a list of the nodes in
    # each community - these nodes are currently just numbers (which are mapped to Pyomo modeling components
    # with number_component_map)

    # Now, we want to include another list for each community - the new list will be specific to the
    # type_of_community_map specified by the user, and is described within the conditionals below

    # Also, as this second list is constructed, the node values will be converted back to the Pyomo components
    # through the use of number_component_map, resulting in a dictionary where the values are two-list tuples that
    # contain Pyomo modeling components

    if type_of_community_map == 'bipartite':
        # If the community map was created for a bipartite graph, then for a given community, we simply want to
        # separate the nodes into their two groups; thus, we create a list of constraints and a list of variables

        for community_key in community_map:
            constraint_node_list, variable_node_list = [], []
            node_community_list = community_map[community_key]
            for numbered_node in node_community_list:
                if numbered_node in constraint_variable_map:
                    constraint_node_list.append(number_component_map[numbered_node])
                else:
                    variable_node_list.append(number_component_map[numbered_node])
            community_map[community_key] = (constraint_node_list, variable_node_list)

    elif type_of_community_map == 'constraint':
        # If the community map was created for a constraint node graph, then for a given community, we want to create a
        # new list that contains all of the variables contained in the constraint equations of that community

        for community_key in community_map:
            constraint_list = sorted(community_map[community_key])
            variable_list = [constraint_variable_map[numbered_constraint] for numbered_constraint in constraint_list]
            variable_list = sorted(set([node for variable_sublist in variable_list for node in variable_sublist]))
            variable_list = [number_component_map[variable] for variable in variable_list]
            constraint_list = [number_component_map[constraint] for constraint in constraint_list]
            community_map[community_key] = (constraint_list, variable_list)

    elif type_of_community_map == 'variable':
        # If the community map was created for a variable node graph, then for a given community, we want to create a
        # new list that contains all of the constraints that the variables of that community appear in

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

    # Thus, each key in community_map now maps to a tuple of two lists, a constraint list and a variable list (in that
    # order)

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


def visualize_model_graph(model, community_map=None, type_of_graph='constraint', with_objective=True,
                          weighted_graph=True,
                          random_seed=None, type_of_community_map=None, pos=None):
    """
    This function draws a graph of the communities for a Pyomo model.

    This function takes in a Pyomo model and its community map - if no community map is given, a community map is
    created with the detect_communities function. A NetworkX graph of the model is created with the function
    generate_model_graph, using the parameters specified by the user. The model and some of the given parameters
    (type_of_graph, with_objective) are used to create the nodes and edges for the model graph illustration. The
    community map is used to color the nodes according to their communities, and if no community map is given,
    then the model and some of the given parameters (type_of_community_map, with_objective, weighted_graph) are
    used in the function detect_communities to create a community map.

    Parameters
    ----------
    model: Block
         a Pyomo model or block to be used for community detection
    community_map: dict, optional
        a dictionary that maps an integer key (which corresponds to a community number) to a tuple of two lists.
        The first list should contain the constraints in the given community and the second list should contain the
        variables in the given community.
    type_of_graph: str, optional
        a string that specifies the types of nodes drawn on the model graph, the default is 'constraint'.
        'constraint' draws a graph with constraint nodes,
        'variable' draws a graph with variable nodes,
        'bipartite' draws a graph with both constraint and variable nodes (bipartite graph).
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function is included in the graph; the
        default is True
    weighted_graph: bool, optional
        (this argument is only used if no community_map is provided)
        a Boolean argument that specifies whether a community map is created based on a weighted graph or an
        unweighted graph; the default is True (type_of_community_map='bipartite' creates an unweighted graph
        regardless of this parameter)
    random_seed: int, optional
        (this argument is only used if no community_map is provided)
        an integer that is used as the random seed for the (heuristic) Louvain community detection
    type_of_community_map: str, optional
        (this argument is only used if no community_map is provided)
        this is used in the function detect_communities to create a community map; the default value is whatever value
        is given to 'type_of_graph'.
        'constraint' creates a community map based on constraint nodes,
        'variable' creates a community map based on variable nodes,
        'bipartite' creates a community map based on constraint and variable nodes (bipartite graph)
    pos: dict, optional
        a dictionary that maps node keys to their positions on the illustration

    Returns
    -------
    fig: matplotlib figure
        the figure for the model graph drawing - can be illustrated by calling 'plt.show()'
    pos: dict
        a dictionary that maps node keys to their positions on the illustration - can be used to create consistent
        layouts for graphs of a given model
    """

    # Check that all arguments are of the correct type

    assert isinstance(model, ConcreteModel), "Invalid model: 'model=%s' - model must be an instance of " \
                                             "ConcreteModel" % model

    assert type_of_graph in (
    'bipartite', 'constraint', 'variable'), "Invalid graph type specified: 'type_of_graph=%s' - Valid " \
                                            "values: 'bipartite', 'constraint', 'variable'" % type_of_graph

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert random_seed is None or (type(random_seed) == int and random_seed >= 0), \
        "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed

    assert type_of_community_map is None or type_of_community_map in ('bipartite', 'constraint', 'variable'), \
        "Invalid value for type_of_community_map: 'type_of_community_map=%s' - Valid values: 'bipartite', 'constraint', 'variable'" \
        % type_of_community_map

    # No assert statement for pos; the NetworkX function can handle issues with pos

    # Use the generate_model_graph function to create a NetworkX graph of the given model (along with
    # number_component_map and constraint_variable_map, which will be used to help with drawing the graph)
    model_graph, number_component_map, constraint_variable_map = generate_model_graph(
        model, type_of_graph=type_of_graph, with_objective=with_objective, weighted_graph=False)

    # This line creates the "reverse" of the number_component_map returned by generate_model_graph above,
    # since component_number_map is more convenient to use in this function
    component_number_map = ComponentMap((comp, number) for number, comp in number_component_map.items())

    if community_map is None:
        user_provided_community_map = False  # Will be used for the graph title

        # The default value for type_of_community_map is the same as type_of_graph (and type_of_community_map is
        # only used when the user does not provide a community_map)
        if type_of_community_map is None:
            type_of_community_map = type_of_graph

        # Since no community map was given by the user, a community map will be created using the given
        # model and the detect_communities function
        community_map = detect_communities(model, type_of_community_map=type_of_community_map,
                                           with_objective=with_objective,
                                           weighted_graph=weighted_graph, random_seed=random_seed)

    else:  # This is the case where the user has provided their own community_map

        if type_of_community_map is not None:  # Preference is given to the user-provided community map
            # type_of_community_map should only be specified if no community_map is being provided
            logger.info("An argument was provided for both 'community_map' and 'type_of_community_map' - only one of "
                        "these two parameters should be specified (see docstring). The given community map was used; "
                        "'type_of_community_map' was not used")

        user_provided_community_map = True  # Will be used for the graph title

        # Check that the contents of the dictionary are of the right types
        assert type(community_map) == dict, "community_map should be a Python dictionary"

        # Note there is no assertion for the dictionary keys - they do not have to be anything specific

        for community_map_value in list(community_map.values()):
            assert len(community_map_value) == 2 and type(community_map_value) == tuple and \
                   type(community_map_value[0]) == list and type(community_map_value[1]) == list, \
                "The values of community_map should all be tuples containing two lists"

        for community_key in community_map:
            tuple_of_lists = community_map[community_key]
            for community_member in (tuple_of_lists[0] + tuple_of_lists[1]):
                assert community_member in component_number_map, \
                    "All of the list items in community_map should be Pyomo components that exist in the given model"

    # Now we will use the component_number_map to change the Pyomo modeling components in community_map into the
    # numbers that correspond to their nodes/edges in the NetworkX graph, model_graph
    for key in community_map:
        community_map[key] = ([component_number_map[component] for component in community_map[key][0]],
                              [component_number_map[component] for component in community_map[key][1]])

    # Based on type_of_graph, which specifies what Pyomo modeling components are to be drawn as nodes in the graph
    # illustration, we will now get the node list and the color list, which describes how to color nodes
    # according to their communities (which is based on community_map)
    if type_of_graph == 'bipartite':
        list_of_node_lists = [list_of_nodes for list_tuple in community_map.values() for list_of_nodes in
                              list_tuple]

        # list_of_node_lists is (as it implies) a list of lists, so we will use the list comprehension
        # below to flatten the list and get our one-dimensional node list
        node_list = [node for sublist in list_of_node_lists for node in sublist]

        color_list = []
        # Now, we will find the first community that a node appears in and color the node based on that community
        # In community_map, certain nodes may appear in multiple communities, and we have chosen to give preference
        # to the first community a node appears in
        for node in node_list:
            not_found = True
            for community_key in community_map:
                if not_found and node in (community_map[community_key][0] + community_map[community_key][1]):
                    color_list.append(community_key)
                    not_found = False

        # Find top_nodes (one of the two "groups" of nodes in a bipartite graph), which will be used to
        # determine the graph layout
        if model_graph.number_of_nodes() > 0 and nx.is_connected(model_graph):
            # An index of 1 used because this tends to place constraint nodes on the left, which is
            # consistent with the else case
            top_nodes = nx.bipartite.sets(model_graph)[1]
        else:
            top_nodes = {node for node in model_graph.nodes() if node in constraint_variable_map}

        if pos is None:  # The case where the user has not provided their own layout
            pos = nx.bipartite_layout(model_graph, top_nodes)

    else:  # This covers the case that type_of_community_map is 'constraint' or 'variable'

        # Constraints are in the first list of the tuples in community map and variables are in the second list
        position = 0 if type_of_graph == 'constraint' else 1
        list_of_node_lists = list(i[position] for i in community_map.values())

        # list_of_node_lists is (as it implies) a list of lists, so we will use the list comprehension
        # below to flatten the list and get our one-dimensional node list
        node_list = [node for sublist in list_of_node_lists for node in sublist]

        # Now, we will find the first community that a node appears in and color the node based on that community
        # In community_map, certain nodes may appear in multiple communities, and we have chosen to give preference
        # to the first community a node appears in
        color_list = []
        for node in node_list:
            not_found = True
            for community_key in community_map:
                if not_found and node in community_map[community_key][position]:
                    color_list.append(community_key)
                    not_found = False

        # Note - there is no strong reason to choose spring layout; it just creates relatively clean graphs
        if pos is None:  # The case where the user has not provided their own layout
            pos = nx.spring_layout(model_graph)

    # Define color_map
    color_map = cm.get_cmap('viridis', len(community_map))

    # Create the figure and draw the graph
    fig = plt.figure()
    nx.draw_networkx_nodes(model_graph, pos, nodelist=node_list, node_size=40, cmap=color_map, node_color=color_list)
    nx.draw_networkx_edges(model_graph, pos, alpha=0.5)

    # Make the title
    node_name_map = {'bipartite': 'Bipartite', 'constraint': 'Constraint', 'variable': 'Variable'}
    graph_type = node_name_map[type_of_graph]
    if user_provided_community_map:
        plot_title = "%s graph - colored using user-provided community map" % graph_type
    else:
        community_map_type = node_name_map[type_of_community_map]
        plot_title = "%s graph - colored using %s community map" % (graph_type, community_map_type)
    plt.title(plot_title)

    # Return the figure and the position dictionary used for the graph layout
    return fig, pos


def stringify_community_map(model=None, community_map=None, type_of_community_map='constraint', with_objective=True,
                            weighted_graph=True, random_seed=None):
    """
    This function takes in a community map of Pyomo components and returns the same community map but with the strings
    of the Pyomo components. Alternatively, this function can take in a model and return a community map
    (using the function detect_communities) of the strings of Pyomo components in the communities.

    Parameters
    ----------
    model: Block, optional
        a Pyomo model or block to be used for community detection (only used if community_map is None)
    community_map: dict, optional
        a dictionary with values that contain Pyomo components which will be converted to their strings
    type_of_community_map: str, optional
        (this argument is only used if no community_map is provided)
        a string that specifies the type of community map to be returned, the default is 'constraint'.
        'constraint' returns a dictionary (community_map) with communities based on constraint nodes,
        'variable' returns a dictionary (community_map) with communities based on variable nodes,
        'bipartite' returns a dictionary (community_map) with communities based on constraint and variable nodes (bipartite
        graph)
    with_objective: bool, optional
        (this argument is only used if no community_map is provided)
        a Boolean argument that specifies whether or not the objective function is
        included in the graph (and thus in the community map); the default is True
    weighted_graph: bool, optional
        (this argument is only used if no community_map is provided)
        a Boolean argument that specifies whether the community map is created based on a weighted model graph or an
        unweighted model graph; the default is True (type_of_community_map='bipartite' creates an unweighted model graph
        regardless of this parameter)
    random_seed: int, optional
        (this argument is only used if no community_map is provided)
        an integer that is used as the random seed for the (heuristic) Louvain community detection

    Returns
    -------
    community_map: dict
        a Python dictionary that maps arbitrary keys (in this case, integers from zero to the number of
        communities minus one) to two-list tuples containing the strings of the Pyomo components in the given community
    """

    # Check that arguments are of the right type

    assert model is not None or community_map is not None, "Either a model or a community map must be given as an input"

    assert isinstance(model, ConcreteModel) or model is None, "Invalid model: 'model=%s' - model must be " \
                                                              "an instance of ConcreteModel or None " \
                                                              "if a community map is given" % model

    assert type_of_community_map in ('bipartite', 'constraint', 'variable'), "Invalid value for type_of_community_map: " \
                                                                             "'type_of_community_map=%s' - Valid values: 'bipartite', 'constraint', 'variable'" \
                                                                             % type_of_community_map

    assert type(with_objective) == bool, "Invalid value for with_objective: 'with_objective=%s' - with_objective " \
                                         "must be a Boolean" % with_objective

    assert type(weighted_graph) == bool, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph " \
                                         "must be a Boolean" % weighted_graph

    assert random_seed is None or (type(random_seed) == int and random_seed >= 0), \
        "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed

    if community_map is None:
        # If no community map is given by the user, then a community map will be created using the given
        # model and the detect_communities function
        community_map = detect_communities(model, type_of_community_map=type_of_community_map,
                                           with_objective=with_objective,
                                           weighted_graph=weighted_graph, random_seed=random_seed)

    else:  # This is the case where the user has provided their own community_map

        # Check that the contents of the dictionary are of the right types
        assert type(community_map) == dict, "community_map should be a Python dictionary"
        # Note that the dictionary keys do not have to be anything specific
        for community_map_value in list(community_map.values()):
            assert len(community_map_value) == 2 and type(community_map_value) == tuple and \
                   type(community_map_value[0]) == list and type(community_map_value[1]) == list, \
                "The values of community_map should all be tuples containing two lists"

    # Convert the components in community_map to their strings
    for key in community_map:
        community_map[key] = ([str(component) for component in community_map[key][0]],
                              [str(component) for component in community_map[key][1]])

    return community_map
