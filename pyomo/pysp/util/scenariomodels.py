#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# grab the pyomo.environ components.
from pyomo.core import *

def scenario_tree_model():

    model = AbstractModel()

    # all set/parameter values are strings, representing the names of various entities/variables.

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
                                         mutable=True)

    model.Scenarios = Set(ordered=True)
    model.ScenarioLeafNode = Param(model.Scenarios,
                                   within=model.Nodes,
                                   mutable=True)

    model.StageVariables = Set(model.Stages,
                               initialize=[],
                               ordered=True)
    model.StageCostVariable = Param(model.Stages,
                                    mutable=True)

    # it is often the case that a subset of the stage variables are strictly "derived"
    # variables, in that their values are computable once the values of other variables
    # in that stage are known. it generally useful to know which variables these are,
    # as it is unnecessary to post non-anticipativity constraints for these variables.
    # further, attempting to force non-anticipativity - either implicitly or explicitly -
    # on these variables can cause issues with decomposition algorithms.
    # NOTE: derived variables must appear in the set of StageVariables, i.e.,
    #       StageDerivedVariables must be a subset (possibly empty) of StageVariables.
    model.StageDerivedVariables = Set(model.Stages,
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
#   - StageCostVariable
#   - StageDerivedVariables
# and optionally:
#   - ScenarioBasedData
#   - Bundling
#   - Bundles
#   - BundleScenarios
#
def generate_simple_twostage(num_scenarios):
    m = scenario_tree_model()
    m.Stages.add('Stage1')
    m.Stages.add('Stage2')
    m.Nodes.add('RootNode')
    for i in range(1, num_scenarios+1):
        m.Nodes.add('LeafNode_Scenario'+str(i))
        m.Scenarios.add('Scenario'+str(i))
    m = m.create_instance()
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
