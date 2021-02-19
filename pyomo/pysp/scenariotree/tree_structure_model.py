#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

__all__ = ()

import pyomo.core

import six

from pyomo.common.dependencies import attempt_import

# The code below conforms to the networkx>=2.0 API
networkx, networkx_available = attempt_import('networkx', minimum_version="2.0")

def CreateAbstractScenarioTreeModel():
    from pyomo.core import (
        AbstractModel, Set, Param, Boolean, Any, UnitInterval,
    )

    model = AbstractModel()

    # all set/parameter values are strings, representing the
    # names of various entities/variables.

    model.Stages = Set(ordered=True)
    model.Nodes = Set(ordered=True)

    model.NodeStage = Param(model.Nodes,
                            within=model.Stages,
                            mutable=True)
    model.Children = Set(model.Nodes,
                         within=model.Nodes,
                         initialize=[],
                         ordered=True)
    model.ConditionalProbability = Param(model.Nodes,
                                         within=UnitInterval,
                                         mutable=True)

    model.Scenarios = Set(ordered=True)
    model.ScenarioLeafNode = Param(model.Scenarios,
                                   within=model.Nodes,
                                   mutable=True)

    model.StageVariables = Set(model.Stages,
                               initialize=[],
                               ordered=True)

    model.NodeVariables = Set(model.Nodes,
                              initialize=[],
                              ordered=True)

    model.StageCost = Param(model.Stages,
                            within=Any,
                            mutable=True,
                            default=None)
    model.NodeCost = Param(model.Nodes,
                           within=Any,
                           mutable=True,
                           default=None)

    # DEPRECATED
    model.StageCostVariable = Param(model.Stages,
                                    within=Any,
                                    mutable=True)

    # it is often the case that a subset of the stage variables are strictly "derived"
    # variables, in that their values are computable once the values of other variables
    # in that stage are known. it generally useful to know which variables these are,
    # as it is unnecessary to post non-anticipativity constraints for these variables.
    # further, attempting to force non-anticipativity - either implicitly or explicitly -
    # on these variables can cause issues with decomposition algorithms.
    model.StageDerivedVariables = Set(model.Stages,
                                      initialize=[],
                                      ordered=True)
    model.NodeDerivedVariables = Set(model.Nodes,
                                     initialize=[],
                                     ordered=True)

    # scenario data can be populated in one of two ways. the first is "scenario-based",
    # in which a single .dat file contains all of the data for each scenario. the .dat
    # file prefix must correspond to the scenario name. the second is "node-based",
    # in which a single .dat file contains only the data for each node in the scenario
    # tree. the node-based method is more compact, but the scenario-based method is
    # often more natural when parameter data is generated via simulation. the default
    # is scenario-based.
    model.ScenarioBasedData = Param(within=Boolean,
                                    default=True,
                                    mutable=True)

    # do we bundle, and if so, how?
    model.Bundling = Param(within=Boolean,
                           default=False,
                           mutable=True)

    # bundle names
    model.Bundles = Set(ordered=True)
    model.BundleScenarios = Set(model.Bundles,
                                ordered=True)

    return model

#
# Generates a simple two-stage scenario tree model with the requested
# number of scenarios. It is up to the user to include the remaining
# data for:
#   - StageVariables
#   - StageCost
# and optionally:
#   - StageDerivedVariables
#   - ScenarioBasedData
#   - Bundling
#   - Bundles
#   - BundleScenarios
#
def CreateConcreteTwoStageScenarioTreeModel(num_scenarios):
    m = CreateAbstractScenarioTreeModel()
    m = m.create_instance()
    m.Stages.add('Stage1')
    m.Stages.add('Stage2')
    m.Nodes.add('RootNode')
    for i in range(1, num_scenarios+1):
        m.Nodes.add('LeafNode_Scenario'+str(i))
        m.Scenarios.add('Scenario'+str(i))
    m.NodeStage['RootNode'] = 'Stage1'
    m.ConditionalProbability['RootNode'] = 1.0
    for node in m.Nodes:
        if node != 'RootNode':
            m.NodeStage[node] = 'Stage2'
            m.Children['RootNode'].add(node)
            m.Children[node].clear()
            m.ConditionalProbability[node] = 1.0/num_scenarios
            m.ScenarioLeafNode[node.replace('LeafNode_','')] = node

    return m

def ScenarioTreeModelFromNetworkX(
        tree,
        node_name_attribute=None,
        edge_probability_attribute='weight',
        stage_names=None,
        scenario_name_attribute=None):
    """
    Create a scenario tree model from a networkx tree.  The
    height of the tree must be at least 1 (meaning at least
    2 stages).

    Required node attributes:
        - cost (str): A string identifying a component on
              the model whose value indicates the cost at
              the time stage of the node for any scenario
              traveling through it.

    Optional node attributes:
        - variables (list): A list of variable identifiers
              that will be tracked by the node. If the node
              is not a leaf node, these indicate variables
              with non-anticipativity constraints.
        - derived_variables (list): A list of variable or
              expression identifiers that will be tracked by
              the node (but will never have
              non-anticipativity constraints enforced).
        - bundle: A bundle identifier for the scenario
              defined by a leaf-stage node. This attribute
              is ignored on non-terminal tree nodes. This
              attribute appears on at least one leaf-stage
              node (and is not set to :const:`None`), then
              it must be set on all leaf-stage nodes (to
              something other than :const:`None`);
              otherwise, an exception will be raised.

    Optional edge attributes:
        - weight (float): Indicates the conditional
              probability of moving from the parent node to
              the child node in the directed edge. If not
              present, it will be assumed that all edges
              leaving the parent node have equal probability
              (normalized to sum to one).

    Args:
        stage_names: Can define a list of stage names to use
           (assumed in time order). The length of this list
           much match the number of stages in the tree. The
           default value of :const:`None` indicates that
           stage names should be automatically generated in
           with the form ['Stage1','Stage2',...].
        node_name_attribute: By default, node names are the
           same as the node hash in the networkx tree. This
           keyword can be set to the name of some property
           of nodes in the graph that will be used for their
           name in the PySP scenario tree.
        scenario_name_attribute: By default, scenario names
           are the same as the leaf-node hash in the
           networkx tree. This keyword can be set to the
           name of some property of leaf-nodes in the graph
           that will be used for their corresponding
           scenario name in the PySP scenario tree.
        edge_probability_attribute: Can be set to the name
           of some property of edges in the graph that
           defines the conditional probability of that
           branch (default: 'weight'). If this keyword is
           set to :const:`None`, then all branches leaving a
           node are assigned equal conditional
           probabilities.

    Examples:

        A 2-stage scenario tree with 10 scenarios grouped
        into 2 bundles:

        >>> G = networkx.DiGraph()
        >>> G.add_node("root", variables=["x"])
        >>> N = 10
        >>> for i in range(N):
        >>>     node_name = "s"+str(i)
        >>>     bundle_name = "b"+str(i%2)
        >>>     G.add_node(node_name, bundle=bundle)
        >>>     G.add_edge("root", node_name, weight=1.0/N)
        >>> model = ScenarioTreeModelFromNetworkX(G)

        A 4-stage scenario tree with 125 scenarios:

        >>> branching_factor = 5
        >>> height = 3
        >>> G = networkx.balanced_tree(
                   branching_factor,
                   height,
                   networkx.DiGraph())
        >>> model = ScenarioTreeModelFromNetworkX(G)
    """

    if not networkx.is_tree(tree):
        raise TypeError(
            "Graph object is not a tree "
            "(see networkx.is_tree)")

    if not networkx.is_directed(tree):
        raise TypeError(
            "Graph object is not directed "
            "(see networkx.is_directed)")

    if not networkx.is_branching(tree):
        raise TypeError(
            "Grapn object is not a branching "
            "(see networkx.is_branching")

    in_degree_items = tree.in_degree()
    # Prior to networkx ~2.0, in_degree() returned a dictionary.
    # Now it is a view on items, so only call .items() for the old case
    if hasattr(in_degree_items, 'items'):
        in_degree_items = in_degree_items.items()
    root = [u for u,d in in_degree_items if d == 0]
    assert len(root) == 1
    root = root[0]
    num_stages = networkx.eccentricity(tree, v=root) + 1
    if num_stages < 2:
        raise ValueError(
            "The number of stages must be at least 2")
    m = CreateAbstractScenarioTreeModel()
    m = m.create_instance()
    if stage_names is not None:
        unique_stage_names = set()
        for cnt, stage_name in enumerate(stage_names,1):
            m.Stages.add(stage_name)
            unique_stage_names.add(stage_name)
        if cnt != num_stages:
            raise ValueError(
                "incorrect number of stages names (%s), should be %s"
                % (cnt, num_stages))
        if len(unique_stage_names) != cnt:
            raise ValueError("all stage names were not unique")
    else:
        for i in range(num_stages):
            m.Stages.add('Stage'+str(i+1))
    node_to_name = {}
    node_to_scenario = {}
    scenario_bundle = {}
    def _setup(u, succ):
        if node_name_attribute is not None:
            if node_name_attribute not in tree.nodes[u]:
                raise KeyError(
                    "node '%s' missing node name "
                    "attribute: '%s'"
                    % (u, node_name_attribute))
            node_name = tree.nodes[u][node_name_attribute]
        else:
            node_name = u
        node_to_name[u] = node_name
        m.Nodes.add(node_name)
        if u in succ:
            for v in succ[u]:
                _setup(v, succ)
        else:
            # a leaf node
            if scenario_name_attribute is not None:
                if scenario_name_attribute not in tree.nodes[u]:
                    raise KeyError(
                        "node '%s' missing scenario name "
                        "attribute: '%s'"
                        % (u, scenario_name_attribute))
                scenario_name = tree.nodes[u][scenario_name_attribute]
            else:
                scenario_name = u
            node_to_scenario[u] = scenario_name
            m.Scenarios.add(scenario_name)
            scenario_bundle[scenario_name] = \
                tree.nodes[u].get('bundle', None)
    _setup(root,
           networkx.dfs_successors(tree, root))

    def _add_node(u, stage, succ, pred):
        node_name = node_to_name[u]
        m.NodeStage[node_name] = m.Stages[stage]
        if u == root:
            m.ConditionalProbability[node_name] = 1.0
        else:
            assert u in pred
            # prior to networkx ~2.0, we used a .edge attribute on DiGraph,
            # which no longer exists.
            if hasattr(tree, 'edge'):
                edge = tree.edge[pred[u]][u]
            else:
                edge = tree.edges[pred[u],u]
            probability = None
            if edge_probability_attribute is not None:
                if edge_probability_attribute not in edge:
                    raise KeyError(
                        "edge '(%s, %s)' missing probability attribute: '%s'"
                        % (pred[u], u, edge_probability_attribute))
                probability = edge[edge_probability_attribute]
            else:
                probability = 1.0/len(succ[pred[u]])
            m.ConditionalProbability[node_name] = probability
        # get node variables
        if "variables" in tree.nodes[u]:
            node_variables = tree.nodes[u]["variables"]
            assert type(node_variables) in [tuple, list]
            for varstring in node_variables:
                m.NodeVariables[node_name].add(varstring)
        if "derived_variables" in tree.nodes[u]:
            node_derived_variables = tree.nodes[u]["derived_variables"]
            assert type(node_derived_variables) in [tuple, list]
            for varstring in node_derived_variables:
                m.NodeDerivedVariables[node_name].add(varstring)
        if "cost" in tree.nodes[u]:
            assert isinstance(tree.nodes[u]["cost"], six.string_types)
            m.NodeCost[node_name].value = tree.nodes[u]["cost"]
        if u in succ:
            child_names = []
            for v in succ[u]:
                child_names.append(
                    _add_node(v, stage+1, succ, pred))
            total_probability = 0.0
            for child_name in child_names:
                m.Children[node_name].add(child_name)
                total_probability += \
                    pyomo.core.value(m.ConditionalProbability[child_name])
            if abs(total_probability - 1.0) > 1e-5:
                raise ValueError(
                    "edge probabilities leaving node '%s' "
                    "do not sum to 1 (total=%r)"
                    % (u, total_probability))
        else:
            # a leaf node
            scenario_name = node_to_scenario[u]
            m.ScenarioLeafNode[scenario_name] = node_name
            m.Children[node_name].clear()

        return node_name

    _add_node(root,
              1,
              networkx.dfs_successors(tree, root),
              networkx.dfs_predecessors(tree, root))

    if any(_b is not None for _b in scenario_bundle.values()):
        if any(_b is None for _b in scenario_bundle.values()):
            raise ValueError("Incomplete bundle specification. "
                             "All scenarios require a bundle "
                             "identifier.")
        m.Bundling.value = True
        bundle_scenarios = {}
        for bundle_name in sorted(set(scenario_bundle.values())):
            m.Bundles.add(bundle_name)
            bundle_scenarios[bundle_name] = []
        for scenario_name in m.Scenarios:
            bundle_scenarios[scenario_bundle[scenario_name]].\
                append(scenario_name)
        for bundle_name in m.Bundles:
            for scenario_name in sorted(bundle_scenarios[bundle_name]):
                m.BundleScenarios[bundle_name].add(scenario_name)

    return m
