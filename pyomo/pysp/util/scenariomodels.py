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

scenario_tree_model = AbstractModel()

# all set/parameter values are strings, representing the names of various entities/variables.

scenario_tree_model.Stages = Set(ordered=True)
scenario_tree_model.Nodes = Set(ordered=True)

scenario_tree_model.NodeStage = Param(scenario_tree_model.Nodes, within=scenario_tree_model.Stages)
scenario_tree_model.Children = Set(scenario_tree_model.Nodes, within=scenario_tree_model.Nodes, ordered=True)
scenario_tree_model.ConditionalProbability = Param(scenario_tree_model.Nodes)

scenario_tree_model.Scenarios = Set(ordered=True)
scenario_tree_model.ScenarioLeafNode = Param(scenario_tree_model.Scenarios, within=scenario_tree_model.Nodes)

scenario_tree_model.StageVariables = Set(scenario_tree_model.Stages, initialize=[], ordered=True)
scenario_tree_model.StageCostVariable = Param(scenario_tree_model.Stages)

# it is often the case that a subset of the stage variables are strictly "derived"
# variables, in that their values are computable once the values of other variables
# in that stage are known. it generally useful to know which variables these are,
# as it is unnecessary to post non-anticipativity constraints for these variables.
# further, attempting to force non-anticipativity - either implicitly or explicitly -
# on these variables can cause issues with decomposition algorithms.
# NOTE: derived variables must appear in the set of StageVariables, i.e., 
#       StageDerivedVariables must be a subset (possibly empty) of StageVariables.
scenario_tree_model.StageDerivedVariables = Set(scenario_tree_model.Stages, initialize=[], ordered=True)

# scenario data can be populated in one of two ways. the first is "scenario-based",
# in which a single .dat file contains all of the data for each scenario. the .dat
# file prefix must correspond to the scenario name. the second is "node-based",
# in which a single .dat file contains only the data for each node in the scenario
# tree. the node-based method is more compact, but the scenario-based method is
# often more natural when parameter data is generated via simulation. the default
# is scenario-based.
scenario_tree_model.ScenarioBasedData = Param(within=Boolean, default=True, mutable=True)

# do we bundle, and if so, how?
scenario_tree_model.Bundling = Param(within=Boolean, default=False, mutable=True)
scenario_tree_model.Bundles = Set(ordered=True) # bundle names
scenario_tree_model.BundleScenarios = Set(scenario_tree_model.Bundles, ordered=True)
