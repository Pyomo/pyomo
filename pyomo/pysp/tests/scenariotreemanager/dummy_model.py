#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import ConcreteModel, Var, Expression, ConstraintList, Objective, sum_product

def pysp_scenario_tree_model_callback():
    from pyomo.pysp.scenariotree.tree_structure_model \
        import CreateConcreteTwoStageScenarioTreeModel

    st_model = CreateConcreteTwoStageScenarioTreeModel(3)

    first_stage = st_model.Stages.first()
    second_stage = st_model.Stages.last()

    # First Stage
    st_model.StageCost[first_stage] = 'StageCost[1]'
    st_model.StageVariables[first_stage].add('x')
    st_model.StageDerivedVariables[first_stage].add('y')

    # Second Stage
    st_model.StageCost[second_stage] = 'StageCost[2]'
    st_model.StageVariables[second_stage].add('z')
    st_model.StageDerivedVariables[second_stage].add('q')

    return st_model

cnt = 0
def pysp_instance_creation_callback(scenario_name, node_names):
    global cnt

    model = ConcreteModel()
    model.x = Var(bounds=(0,10))
    model.y = Expression(expr=model.x + 1)
    model.z = Var(bounds=(-10, 10))
    model.q = Expression(expr=model.z**2)
    model.StageCost = Expression([1,2])
    model.StageCost.add(1, model.x)
    model.StageCost.add(2, -model.z)
    model.o = Objective(expr=sum_product(model.StageCost))
    model.c = ConstraintList()
    model.c.add(model.x >= cnt)
    model.c.add(model.z <= cnt**2)

    cnt += 1

    return model
