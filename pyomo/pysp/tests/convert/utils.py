#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import *
from pyomo.pysp.scenariotree.tree_structure_model import \
    ScenarioTreeModelFromNetworkX
from pyomo.pysp.annotations import (ConstraintStageAnnotation,
                                    StochasticConstraintBoundsAnnotation,
                                    StochasticConstraintBodyAnnotation,
                                    StochasticObjectiveAnnotation,
                                    StochasticVariableBoundsAnnotation)

def simple_twostage_scenario_tree():
    from pyomo.pysp.scenariotree.tree_structure_model \
        import CreateConcreteTwoStageScenarioTreeModel
    st_model = CreateConcreteTwoStageScenarioTreeModel(2)
    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()
    # First Stage
    st_model.StageCost[first_stage] = 'StageCost[1]'
    st_model.StageVariables[first_stage].add('x')
    # Second Stage
    st_model.StageCost[second_stage] = 'StageCost[2]'
    st_model.StageVariables[second_stage].add('y')
    return st_model

def simple_twostage_model():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.StageCost = Expression([1,2])
    model.StageCost.add(1, model.x)
    model.StageCost.add(2, model.y)
    model.c = ConstraintList()
    model.c.add(model.x >= 10)
    model.c.add(model.y >= 10)
    model.o = Objective(expr=sum_product(model.StageCost))
    return model

def simple_threestage_scenario_tree():
    from pyomo.pysp.scenariotree.tree_structure_model \
        import CreateConcreteTwoStageScenarioTreeModel
    import networkx
    G = networkx.balanced_tree(2,2,networkx.DiGraph())
    st_model = ScenarioTreeModelFromNetworkX(
        G,
        edge_probability_attribute=None)
    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.next(first_stage)
    third_stage = st_model.Stages.last()
    # First Stage
    st_model.StageCost[first_stage] = 'StageCost[1]'
    st_model.StageVariables[first_stage].add('x')
    # Second Stage
    st_model.StageCost[second_stage] = 'StageCost[2]'
    st_model.StageVariables[second_stage].add('y')
    # Third Stage
    st_model.StageCost[third_stage] = 'StageCost[3]'
    st_model.StageVariables[second_stage].add('z')
    return st_model

def simple_threestage_model():
    model = ConcreteModel()
    model.x = Var()
    model.y = Var()
    model.z = Var()
    model.StageCost = Expression([1,2,3])
    model.StageCost(1, model.x)
    model.StageCost(2, model.y)
    model.StageCost(3, model.z)
    model.c = ConstraintList()
    model.c.add(model.x >= 10)
    model.c.add(model.y >= 10)
    model.c.add(model.z >= 10)
    model.o = Objective(expr=sum_product(model.StageCost))
    return model
