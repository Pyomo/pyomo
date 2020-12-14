#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Set, ConstraintList, Block
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

def inactive_index_LP_obj_rule(model,i):
    if i == 1:
        return model.x-model.y
    else:
        return -model.x+model.y+model.z

def inactive_index_LP_c2_rule(model,i):
    if i == 1:
        return model.y >= -2
    else:
        return model.x <= 2

@register_model
class LP_inactive_index(_BaseTestModel):
    """
    A continuous linear model where component subindices have been deactivated
    """

    description = "LP_inactive_index"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.s = Set(initialize=[1,2])
        model.x = Var()
        model.y = Var()
        model.z = Var(bounds=(0,None))

        model.obj = Objective(model.s,
                              rule=inactive_index_LP_obj_rule)
        model.OBJ = Objective(expr=model.x+model.y)
        model.obj[1].deactivate()
        model.OBJ.deactivate()
        model.c1 = ConstraintList()
        model.c1.add(model.x<=1)   # index=1
        model.c1.add(model.x>=-1)  # index=2
        model.c1.add(model.y<=1)   # index=3
        model.c1.add(model.y>=-1)  # index=4
        model.c1[1].deactivate()
        model.c1[4].deactivate()
        model.c2 = Constraint(model.s,
                              rule=inactive_index_LP_c2_rule)

        model.b = Block()
        model.b.c = Constraint(expr=model.z >= 2)
        model.B = Block(model.s)
        model.B[1].c = Constraint(expr=model.z >= 3)
        model.B[2].c = Constraint(expr=model.z >= 1)

        model.b.deactivate()
        model.B.deactivate()
        model.B[2].activate()

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None
        model.z.value = 2.0

@register_model
class LP_inactive_index_kernel(LP_inactive_index):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.s = [1,2]
        model.x = pmo.variable()
        model.y = pmo.variable()
        model.z = pmo.variable(lb=0)

        model.obj = pmo.objective_dict()
        for i in model.s:
            model.obj[i] = pmo.objective(
                inactive_index_LP_obj_rule(model,i))

        model.OBJ = pmo.objective(model.x+model.y)
        model.obj[1].deactivate()
        model.OBJ.deactivate()
        model.c1 = pmo.constraint_dict()
        model.c1[1] = pmo.constraint(model.x<=1)
        model.c1[2] = pmo.constraint(model.x>=-1)
        model.c1[3] = pmo.constraint(model.y<=1)
        model.c1[4] = pmo.constraint(model.y>=-1)
        model.c1[1].deactivate()
        model.c1[4].deactivate()
        model.c2 = pmo.constraint_dict()
        for i in model.s:
            model.c2[i] = pmo.constraint(
                inactive_index_LP_c2_rule(model, i))

        model.b = pmo.block()
        model.b.c = pmo.constraint(model.z >= 2)
        model.B = pmo.block_dict()
        model.B[1] = pmo.block()
        model.B[1].c = pmo.constraint(model.z >= 3)
        model.B[2] = pmo.block()
        model.B[2].c = pmo.constraint(model.z >= 1)

        model.b.deactivate()
        model.B[1].deactivate()
