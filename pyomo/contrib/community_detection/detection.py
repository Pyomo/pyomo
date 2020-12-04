"""
Main module for community detection integration with Pyomo models.

This module separates model components (variables, constraints, and objectives) into different communities
distinguished by the degree of connectivity between community members.

Original implementation developed by Rahul Joglekar in the Grossmann research group.

"""
from logging import getLogger

from pyomo.common.dependencies import attempt_import
from pyomo.core import ConcreteModel, ComponentMap, Block, Var, Constraint, Objective, ConstraintList
from pyomo.core.base.objective import _GeneralObjectiveData
from pyomo.core.expr.current import identify_variables
from pyomo.core.expr.visitor import replace_expressions
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.common.dependencies import networkx as nx
from pyomo.common.dependencies import matplotlib, matplotlib_available
from itertools import combinations

import copy

logger = getLogger('pyomo.contrib.community_detection')

# Attempt import of louvain community detection package
community_louvain, community_louvain_available = attempt_import(
    'community', error_message="Could not import the 'community' library, available via 'python-louvain' on PyPI.")

# Import matplotlib
plt = matplotlib.pyplot


def detect_communities(model, type_of_community_map='constraint', with_objective=True, weighted_graph=True,
                       random_seed=None, use_only_active_components=True):
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
        'bipartite' returns a dictionary (community_map) with communities based on a bipartite graph (both constraint
        and variable nodes)
    with_objective: bool, optional
        a Boolean argument that specifies whether or not the objective function is
        included in the model graph (and thus in 'community_map'); the default is True
    weighted_graph: bool, optional
        a Boolean argument that specifies whether community_map is created based on a weighted model graph or an
        unweighted model graph; the default is True (type_of_community_map='bipartite' creates an unweighted
        model graph regardless of this parameter)
    random_seed: int, optional
        an integer that is used as the random seed for the (heuristic) Louvain community detection
    use_only_active_components: bool, optional
        a Boolean argument that specifies whether inactive constraints/objectives are included in the community map

    Returns
    -------
    CommunityMap object (dict-like object)
        The CommunityMap object acts as a Python dictionary, mapping integer keys to tuples containing two lists
        (which contain the components in the given community) - a constraint list and variable list. Furthermore,
        the CommunityMap object stores relevant information about the given community map (dict), such as the model
        used to create it, its networkX representation, etc.
    """

    # Check that all arguments are of the correct type
    if not isinstance(model, ConcreteModel):
        raise TypeError("Invalid model: 'model=%s' - model must be an instance of ConcreteModel" % model)

    if type_of_community_map not in ('bipartite', 'constraint', 'variable'):
        raise TypeError(
            "Invalid value for type_of_community_map: 'type_of_community_map=%s' - "
            "Valid values: 'bipartite', 'constraint', 'variable'" % type_of_community_map)

    if type(with_objective) != bool:
        raise TypeError(
            "Invalid value for with_objective: 'with_objective=%s' - with_objective must be a Boolean" % with_objective)

    if type(weighted_graph) != bool:
        raise TypeError(
            "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph must be a Boolean" % weighted_graph)

    if random_seed is not None:
        if type(random_seed) != int:
            raise TypeError(
                "Invalid value for random_seed: 'random_seed=%s' - "
                "random_seed must be a non-negative integer" % random_seed)
        if random_seed < 0:
            raise ValueError(
                "Invalid value for random_seed: 'random_seed=%s' - "
                "random_seed must be a non-negative integer" % random_seed)

    if use_only_active_components is not True and use_only_active_components is not None:
        raise TypeError(
            "Invalid value for use_only_active_components: 'use_only_active_components=%s' - "
            "use_only_active_components must be True or None" % use_only_active_components)

    # Generate model_graph (a NetworkX graph based on the given Pyomo optimization model),
    # number_component_map (a dictionary to convert the communities into lists of Pyomo components
    # instead of number values), and constraint_variable_map (a dictionary that maps a constraint to the variables
    # it contains)
    model_graph, number_component_map, constraint_variable_map = generate_model_graph(
        model, type_of_graph=type_of_community_map, with_objective=with_objective, weighted_graph=weighted_graph,
        use_only_active_components=use_only_active_components)

    # # TODO - Add option for other community detection package
    # # Maybe something like this:
    # if community_detection_package is not None:
    #     partition_of_graph = community_detection_package(model_graph)

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

    # Return an instance of CommunityMap class which contains the community_map along with other relevant information
    # for the community_map
    return CommunityMap(community_map, type_of_community_map, with_objective, weighted_graph, random_seed,
                        use_only_active_components, model, model_graph, number_component_map, constraint_variable_map,
                        partition_of_graph)


class CommunityMap(object):
    """
    This class is used to create CommunityMap objects which are returned by the detect_communities function. Instances
    of this class allow dict-like usage and store relevant information about the given community map, such as the
    model used to create them, their networkX representation, etc.

    The CommunityMap object acts as a Python dictionary, mapping integer keys to tuples containing two lists
    (which contain the components in the given community) - a constraint list and variable list.

    Methods:
    generate_structured_model
    visualize_model_graph
    """

    def __init__(self, community_map, type_of_community_map, with_objective, weighted_graph, random_seed,
                 use_only_active_components, model, graph, graph_node_mapping, constraint_variable_map,
                 graph_partition):
        """
        Constructor method for the CommunityMap class

        Parameters
        ----------
        community_map: dict
            a Python dictionary that maps arbitrary keys (in this case, integers from zero to the number of
            communities minus one) to two-list tuples containing Pyomo components in the given community
        type_of_community_map: str
            a string that specifies the type of community map to be returned, the default is 'constraint'.
            'constraint' returns a dictionary (community_map) with communities based on constraint nodes,
            'variable' returns a dictionary (community_map) with communities based on variable nodes,
            'bipartite' returns a dictionary (community_map) with communities based on a bipartite graph (both constraint
            and variable nodes)
        with_objective: bool
            a Boolean argument that specifies whether or not the objective function is
            included in the model graph (and thus in 'community_map'); the default is True
        weighted_graph: bool
            a Boolean argument that specifies whether community_map is created based on a weighted model graph or an
            unweighted model graph; the default is True (type_of_community_map='bipartite' creates an unweighted
            model graph regardless of this parameter)
        random_seed: int or None
            an integer that is used as the random seed for the (heuristic) Louvain community detection
        use_only_active_components: bool, optional
            a Boolean argument that specifies whether inactive constraints/objectives are included in the community map
        model: Block
            a Pyomo model or block to be used for community detection
        graph: nx.Graph
            a NetworkX graph with nodes and edges based on the Pyomo optimization model
        graph_node_mapping: dict
            a dictionary that maps a number (which corresponds to a node in the networkX graph representation of the
            model) to a component in the model
        constraint_variable_map: dict
            a dictionary that maps a numbered constraint to a list of (numbered) variables that appear in the constraint
        graph_partition: dict
            the partition of the networkX model graph based on the Louvain community detection
        """

        self.community_map = community_map
        self.type_of_community_map = type_of_community_map
        self.with_objective = with_objective
        self.weighted_graph = weighted_graph
        self.random_seed = random_seed
        self.use_only_active_components = use_only_active_components
        self.model = model
        self.graph = graph
        self.graph_node_mapping = graph_node_mapping
        self.constraint_variable_map = constraint_variable_map
        self.graph_partition = graph_partition

    def __repr__(self):
        """

        repr method changed to return the community_map with the memory locations of the Pyomo components - use str
        method if the strings of the components are desired

        """
        return str(self.community_map)

    def __str__(self):
        """

        str method changed to return the community_map with the strings of the Pyomo components (user-friendly output)

        """
        # Create str_community_map and give it values that are the strings of the components in community_map
        str_community_map = dict()
        for key in self.community_map:
            str_community_map[key] = ([str(component) for component in self.community_map[key][0]],
                                      [str(component) for component in self.community_map[key][1]])

        # Return str_community_map, which is identical to community_map except it has the strings of all of the Pyomo
        # components instead of the actual components
        return str(str_community_map)

    def __eq__(self, other):
        if isinstance(other, dict):
            return self.community_map == other
        elif isinstance(other, CommunityMap):
            return self.community_map == other.community_map
            # Should you check anything else for equality between instances?
        return False

    def __iter__(self):
        for key in self.community_map:
            yield key

    def __getitem__(self, item):
        return self.community_map[item]

    def __len__(self):
        return len(self.community_map)

    def keys(self):
        return self.community_map.keys()

    def values(self):
        return self.community_map.values()

    def items(self):
        return self.community_map.items()

    def visualize_model_graph(self, type_of_graph='constraint', filename=None, pos=None):
        """
        This function draws a graph of the communities for a Pyomo model.

        The type_of_graph parameter is used to create either a variable-node graph, constraint-node graph, or
        bipartite graph of the Pyomo model. Then, the nodes are colored based on the communities they are in - which
        is based on the community map (self.community_map). A filename can be provided to save the figure, otherwise
        the figure is illustrated with matplotlib.

        Parameters
        ----------
        type_of_graph: str, optional
            a string that specifies the types of nodes drawn on the model graph, the default is 'constraint'.
            'constraint' draws a graph with constraint nodes,
            'variable' draws a graph with variable nodes,
            'bipartite' draws a bipartite graph (with both constraint and variable nodes)
        filename: str, optional
            a string that specifies a path for the model graph illustration to be saved
        pos: dict, optional
            a dictionary that maps node keys to their positions on the illustration

        Returns
        -------
        fig: matplotlib figure
            the figure for the model graph drawing
        pos: dict
            a dictionary that maps node keys to their positions on the illustration - can be used to create consistent
            layouts for graphs of a given model
        """

        # Check that all arguments are of the correct type

        assert type_of_graph in ('bipartite', 'constraint', 'variable'), \
            "Invalid graph type specified: 'type_of_graph=%s' - Valid values: " \
            "'bipartite', 'constraint', 'variable'" % type_of_graph

        assert isinstance(filename, (type(None), str)), "Invalid value for filename: 'filename=%s' - filename " \
                                                        "must be a string" % filename

        # No assert statement for pos; the NetworkX function can handle issues with the pos argument

        # There is a possibility that the desired networkX graph of the model is already stored in the
        # CommunityMap object (because the networkX graph is required to create the CommunityMap object)
        if type_of_graph != self.type_of_community_map:
            # Use the generate_model_graph function to create a NetworkX graph of the given model (along with
            # number_component_map and constraint_variable_map, which will be used to help with drawing the graph)
            model_graph, number_component_map, constraint_variable_map = generate_model_graph(
                self.model, type_of_graph=type_of_graph, with_objective=self.with_objective,
                weighted_graph=self.weighted_graph, use_only_active_components=self.use_only_active_components)
        else:
            # This is the case where, as mentioned above, we can use the networkX graph that was made to create
            # the CommunityMap object
            model_graph, number_component_map, constraint_variable_map = self.graph, self.graph_node_mapping, \
                                                                         self.constraint_variable_map

        # This line creates the "reverse" of the number_component_map above, since mapping the Pyomo
        # components to their nodes in the networkX graph is more convenient in this function
        component_number_map = ComponentMap((comp, number) for number, comp in number_component_map.items())

        # Create a deep copy of the community_map attribute to avoid destructively modifying it
        numbered_community_map = copy.deepcopy(self.community_map)

        # Now we will use the component_number_map to change the Pyomo modeling components in community_map into the
        # numbers that correspond to their nodes/edges in the NetworkX graph, model_graph
        for key in self.community_map:
            numbered_community_map[key] = (
                [component_number_map[component] for component in self.community_map[key][0]],
                [component_number_map[component] for component in self.community_map[key][1]])

        # Based on type_of_graph, which specifies what Pyomo modeling components are to be drawn as nodes in the graph
        # illustration, we will now get the node list and the color list, which describes how to color nodes
        # according to their communities (which is based on community_map)
        if type_of_graph == 'bipartite':
            list_of_node_lists = [list_of_nodes for list_tuple in numbered_community_map.values() for list_of_nodes in
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
                for community_key in numbered_community_map:
                    if not_found and node in (
                            numbered_community_map[community_key][0] + numbered_community_map[community_key][1]):
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
            list_of_node_lists = list(i[position] for i in numbered_community_map.values())

            # list_of_node_lists is (as it implies) a list of lists, so we will use the list comprehension
            # below to flatten the list and get our one-dimensional node list
            node_list = [node for sublist in list_of_node_lists for node in sublist]

            # Now, we will find the first community that a node appears in and color the node based on
            # that community (in numbered_community_map, certain nodes may appear in multiple communities,
            # and we have chosen to give preference to the first community a node appears in)
            color_list = []
            for node in node_list:
                not_found = True
                for community_key in numbered_community_map:
                    if not_found and node in numbered_community_map[community_key][position]:
                        color_list.append(community_key)
                        not_found = False

            # Note - there is no strong reason to choose spring layout; it just creates relatively clean graphs
            if pos is None:  # The case where the user has not provided their own layout
                pos = nx.spring_layout(model_graph)

        # Define color_map
        color_map = plt.cm.get_cmap('viridis', len(numbered_community_map))

        # Create the figure and draw the graph
        fig = plt.figure()
        nx.draw_networkx_nodes(model_graph, pos, nodelist=node_list, node_size=40, cmap=color_map,
                               node_color=color_list)
        nx.draw_networkx_edges(model_graph, pos, alpha=0.5)

        # Make the main title
        graph_type = type_of_graph.capitalize()
        community_map_type = self.type_of_community_map.capitalize()
        main_graph_title = "%s graph - colored using %s community map" % (graph_type, community_map_type)

        main_font_size = 14
        plt.suptitle(main_graph_title, fontsize=main_font_size)

        # Define a dict that will be used for the graph subtitle
        subtitle_naming_dict = {
            'bipartite': 'Nodes are variables and constraints & Edges are variables in a constraint',
            'constraint': 'Nodes are constraints & Edges are common variables',
            'variable': 'Nodes are variables & Edges are shared constraints'}

        # Make the subtitle
        subtitle_font_size = 11
        plt.title(subtitle_naming_dict[type_of_graph], fontsize=subtitle_font_size)

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

        # Return the figure and pos, the position dictionary used for the graph layout
        return fig, pos

    def generate_structured_model(self):
        """
        Using the community map and the original model used to create this community map, we will create
        structured_model, which will be based on the original model but will place variables, constraints, and
        objectives into or outside of various blocks (communities) based on the community map.

        Returns
        -------
        structured_model: Block
            a Pyomo model that reflects the nature of the community map
        """

        # Initialize a new model (structured_model) which will contain variables and constraints in blocks based on
        # their respective communities within the CommunityMap
        structured_model = ConcreteModel()

        # Create N blocks (where N is the number of communities found within the model)
        structured_model.b = Block([0, len(self.community_map) - 1, 1])  # values given for (start, stop, step)

        # Initialize a ComponentMap that will map a variable from the model (for example, old_model.x1) used to
        # create the CommunityMap to a list of variables in various blocks that were created based on this
        # variable (for example, [structured_model.b[0].x1, structured_model.b[3].x1])
        blocked_variable_map = ComponentMap()
        # Example key-value pair -> {original_model.x1 : [structured_model.b[0].x1, structured_model.b[3].x1]}

        # TODO - Consider changing structure of the next two for loops to be more efficient (maybe loop through
        #  constraints and add variables as you go) (but note that disconnected variables would be
        #  missed with this strategy)

        # First loop through community_map to add all the variables to structured_model before we add constraints
        # that use those variables
        for community_key, community in self.community_map.items():
            _, variables_in_community = community

            # Loop through all of the variables (from the original model) in the given community
            for stored_variable in variables_in_community:
                # Construct a new_variable whose attributes are determined by querying the variable from the
                # original model
                new_variable = Var(domain=stored_variable.domain, bounds=stored_variable.bounds)

                # Add this new_variable to its block/community and name it using the string of the variable from the
                # original model
                structured_model.b[community_key].add_component(str(stored_variable), new_variable)

                # Since there could be multiple variables 'x1' (such as
                # structured_model.b[0].x1, structured_model.b[3].x1, etc), we need to create equality constraints
                # for all of the variables 'x1' within structured_model (this is the purpose of blocked_variable_map)

                # Here we update blocked_variable_map to keep track of what equality constraints need to be made
                variable_in_new_model = structured_model.find_component(new_variable)
                blocked_variable_map[stored_variable] = blocked_variable_map.get(stored_variable,
                                                                                 []) + [variable_in_new_model]

        # Now that we have all of our variables within the model, we will initialize a dictionary that used to
        # replace variables within constraints to other variables (in our case, this will convert variables from the
        # original model into variables from the new model (structured_model))
        replace_variables_in_expression_map = dict()

        # Loop through community_map again, this time to add constraints (with replaced variables)
        for community_key, community in self.community_map.items():
            constraints_in_community, _ = community

            # Loop through all of the constraints (from the original model) in the given community
            for stored_constraint in constraints_in_community:

                # Now, loop through all of the variables within the given constraint expression
                for variable_in_stored_constraint in identify_variables(stored_constraint.expr):

                    # Loop through each of the "blocked" variables that a variable is mapped to and update
                    # replace_variables_in_expression_map if a variable has a "blocked" form in the given community

                    # What this means is that if we are looping through constraints in community 0, then it would be
                    # best to change a variable x1 into b[0].x1 as opposed to b[2].x1 or b[5].x1 (assuming all of these
                    # blocked versions of the variable x1 exist (which depends on the community map))

                    variable_in_current_block = False
                    for blocked_variable in blocked_variable_map[variable_in_stored_constraint]:
                        if 'b[%d]' % community_key in str(blocked_variable):
                            # Update replace_variables_in_expression_map accordingly
                            replace_variables_in_expression_map[id(variable_in_stored_constraint)] = blocked_variable
                            variable_in_current_block = True

                    if not variable_in_current_block:
                        # Create a version of the given variable outside of blocks then add it to
                        # replace_variables_in_expression_map

                        new_variable = Var(domain=variable_in_stored_constraint.domain,
                                           bounds=variable_in_stored_constraint.bounds)

                        # Add the new variable just as we did above (but now it is not in any blocks)
                        structured_model.add_component(str(variable_in_stored_constraint), new_variable)

                        # Update blocked_variable_map to keep track of what equality constraints need to be made
                        variable_in_new_model = structured_model.find_component(new_variable)
                        blocked_variable_map[variable_in_stored_constraint] = blocked_variable_map.get(
                            variable_in_stored_constraint, []) + [variable_in_new_model]

                        # Update replace_variables_in_expression_map accordingly
                        replace_variables_in_expression_map[id(variable_in_stored_constraint)] = variable_in_new_model

                # TODO - Is there a better way to check whether something is actually an objective? (as done below)
                # Check to see whether 'stored_constraint' is actually an objective (since constraints and objectives
                # grouped together)
                if self.with_objective and isinstance(stored_constraint, (_GeneralObjectiveData, Objective)):
                    # If the constraint is actually an objective, we add it to the block as an objective
                    new_objective = Objective(
                        expr=replace_expressions(stored_constraint.expr, replace_variables_in_expression_map))
                    structured_model.b[community_key].add_component(str(stored_constraint), new_objective)

                else:
                    # Construct a constraint based on the expression within stored_constraint and the dict we have
                    # created for the purpose of replacing the variables within the constraint expression
                    new_constraint = Constraint(
                        expr=replace_expressions(stored_constraint.expr, replace_variables_in_expression_map))

                    # Add this new constraint to the corresponding community/block with its name as the string of the
                    # constraint from the original model
                    structured_model.b[community_key].add_component(str(stored_constraint), new_constraint)

        # If with_objective was set to False, that means we might have missed an objective function within the
        # original model
        if not self.with_objective:
            # Construct a new dictionary for replacing the variables (replace_variables_in_objective_map) which will
            # be specific to the variables in the objective function, since there is the possibility that the
            # objective contains variables we have not yet seen (and thus not yet added to our new model)
            for objective_function in self.model.component_data_objects(ctype=Objective,
                                                                        active=self.use_only_active_components,
                                                                        descend_into=True):

                for variable_in_objective in identify_variables(objective_function):
                    # Add all of the variables in the objective function (not within any blocks)

                    # Check to make sure a form of the variable has not already been made outside of the blocks
                    if structured_model.find_component(str(variable_in_objective)) is None:

                        new_variable = Var(domain=variable_in_objective.domain, bounds=variable_in_objective.bounds)
                        structured_model.add_component(str(variable_in_objective), new_variable)

                        # Again we update blocked_variable_map to keep track of what
                        # equality constraints need to be made
                        variable_in_new_model = structured_model.find_component(new_variable)
                        blocked_variable_map[variable_in_objective] = blocked_variable_map.get(
                            variable_in_objective, []) + [variable_in_new_model]

                        # Update the dictionary that we will use to replace the variables
                        replace_variables_in_expression_map[id(variable_in_objective)] = variable_in_new_model

                    else:
                        for version_of_variable in blocked_variable_map[variable_in_objective]:
                            if 'b[' not in str(version_of_variable):
                                replace_variables_in_expression_map[id(variable_in_objective)] = version_of_variable

                # Now we will construct a new objective function based on the one from the original model and then
                # add it to the new model just as we have done before
                new_objective = Objective(
                    expr=replace_expressions(objective_function.expr, replace_variables_in_expression_map))
                structured_model.add_component(str(objective_function), new_objective)

        # Now, we need to create equality constraints for all of the different "versions" of a variable (such
        # as x1, b[0].x1, b[2].x2, etc.)

        # Create a constraint list for the equality constraints
        structured_model.equality_constraint_list = ConstraintList(doc="Equality Constraints for the different "
                                                                       "forms of a given variable")

        # Loop through blocked_variable_map and create constraints accordingly
        for variable, duplicate_variables in blocked_variable_map.items():
            # variable -> variable from the original model
            # duplicate_variables -> list of variables in the new model

            # Create a list of all the possible equality constraints that need to be made
            equalities_to_make = combinations(duplicate_variables, 2)

            # Loop through the list of two-variable tuples and create an equality constraint for those two variables
            for variable_1, variable_2 in equalities_to_make:
                structured_model.equality_constraint_list.add(expr=variable_1 == variable_2)

        # Return 'structured_model', which is essentially identical to the original model but now has all of the
        # variables, constraints, and objectives placed into blocks based on the nature of the CommunityMap

        return structured_model
