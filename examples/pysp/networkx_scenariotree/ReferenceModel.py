#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

# This PySP example shows how to generate a multi-stage
# scenario tree model from a networkx directed graph.
# The existence of the pysp_scenario_tree_model_callback in
# the model file indicates to PySP that no separate scenario
# tree structure file is required (e.g., ScenarioStructure.dat).
#
# In addition to using networkx, this example shows how
# to declare variables on a per-node basis. Declaring
# variables per-node allows a user to define an SP
# such that the set of variables on a model changes
# across nodes at the same time stage.

from pyomo.environ import *

import networkx

#
# This callback defines a scenario tree model by first
# constructing a networkx directed graph and then calling a
# PySP utility function to convert it to a scenario tree
# model. Once the scenario tree model is created, the
# StageCost, (Node)StageVariable sets are populated with
# variable string templates.
#
def pysp_scenario_tree_model_callback():
    from pyomo.pysp.scenariotree.tree_structure_model import \
        ScenarioTreeModelFromNetworkX

    #
    # Define a tree with three stages using
    # a networkx directed graph
    #
    G = networkx.DiGraph()
    G.add_node("R")
    G.add_node("u0")
    G.add_edge("R", "u0", probability=0.1)
    G.add_node("u1")
    G.add_edge("R", "u1", probability=0.5)
    G.add_node("u2")
    G.add_edge("R", "u2", probability=0.4)
    G.add_node("u00")
    G.add_edge("u0", "u00", probability=0.1)
    G.add_node("u01")
    G.add_edge("u0", "u01", probability=0.9)
    G.add_node("u10")
    G.add_edge("u1", "u10", probability=0.5)
    G.add_node("u11")
    G.add_edge("u1", "u11", probability=0.5)
    G.add_node("u20")
    G.add_edge("u2", "u20", probability=1.0)

    stm = ScenarioTreeModelFromNetworkX(
        G,
        edge_probability_attribute="probability",
        stage_names=["T1", "T2", "T3"])

    # Declare the variables for each node (or stage)
    stm.StageVariables["T1"].add("x")
    stm.StageDerivedVariables["T1"].add("z")
    # for this example, variables in the second and
    # third time-stage change for each node
    stm.NodeVariables["u0"].add("y0")
    stm.NodeDerivedVariables["u0"].add("xu0")
    stm.NodeVariables["u1"].add("y1")
    stm.NodeDerivedVariables["u1"].add("xu1")
    stm.NodeVariables["u2"].add("y2")
    stm.NodeDerivedVariables["u2"].add("xu2")
    stm.NodeVariables["u00"].add("yu00")
    stm.NodeVariables["u01"].add("yu01")
    stm.NodeVariables["u10"].add("yu10")
    stm.NodeVariables["u11"].add("yu11")
    stm.NodeVariables["u20"].add("yu20")

    # Declare the Var or Expression object that
    # reports the cost at each time stage
    stm.StageCost["T1"] = "FirstStageCost"
    stm.StageCost["T2"] = "SecondStageCost"
    stm.StageCost["T3"] = "ThirdStageCost"

    return stm

# Creates an instance for each scenario
def pysp_instance_creation_callback(scenario_name, node_names):

    model = ConcreteModel()
    model.x = Var()
    model.z = Var()
    model.FirstStageCost = Expression(expr=5*(model.z**2 + (model.x-1.1)**2))
    model.SecondStageCost = Expression(expr=0.0)
    model.ThirdStageCost = Expression(expr=0.0)
    model.obj = Objective(expr= model.FirstStageCost + \
                                model.SecondStageCost + \
                                model.ThirdStageCost)
    model.c = ConstraintList()
    model.c.add(model.z == model.x)
    if scenario_name.startswith("u0"):
        # All scenarios under second-stage node "u0"
        model.xu0 = Var()
        model.c.add(model.xu0 == model.x)
        model.SecondStageCost.expr = (model.xu0 - 1)**2

        model.y0 = Var()
        model.c.add(expr= -10 <= model.y0 <= 10)
        if scenario_name == "u00":
            model.yu00 = Var()
            model.c.add(model.yu00 == model.y0)
            model.ThirdStageCost.expr = (model.yu00 + 1)**2
        elif scenario_name == "u01":
            model.yu01 = Var()
            model.c.add(model.yu01 == model.y0)
            model.ThirdStageCost.expr = (2*model.yu01 - 3)**2 + 1
        else:
            assert False
    elif scenario_name.startswith("u1"):
        # All scenarios under second-stage node "u1"
        model.xu1 = Var()
        model.c.add(model.xu1 == model.x)
        model.SecondStageCost.expr = (model.xu1 + 1)**2

        model.y1 = Var()
        model.c.add(expr= -10 <= model.y1 <= 10)
        if scenario_name == "u10":
            model.yu10 = Var()
            model.c.add(model.yu10 == model.y1)
            model.ThirdStageCost.expr = (0.5*model.yu10 - 1)**2 - 1
        elif scenario_name == "u11":
            model.yu11 = Var()
            model.c.add(model.yu11 == model.y1)
            model.ThirdStageCost.expr = (0.2*model.yu11)**2
        else:
            assert False
    elif scenario_name.startswith("u2"):
        # All scenarios under second-stage node "u2"
        model.xu2 = Var()
        model.c.add(model.xu2 == model.x)
        model.SecondStageCost.expr = (model.xu2 - 0.5)**2

        model.y2 = Var()
        model.c.add(expr= -10 <= model.y2 <= 10)
        if scenario_name == "u20":
            model.yu20 = Var()
            model.c.add(model.yu20 == model.y2)
            model.ThirdStageCost.expr = (0.1*model.yu20 - 3)**2 + 2
        else:
            assert False
    else:
        assert False

    return model
