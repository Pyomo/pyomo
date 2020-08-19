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
from pyomo.core import ConcreteModel, Var, Objective, Constraint, Binary, Integers
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class MILP_discrete_var_bounds(_BaseTestModel):
    """
    A discrete model where discrete variables have custom bounds
    """

    description = "MILP_discrete_var_bounds"
    capabilities = set(['linear', 'integer'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.disable_suffix_tests = True
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.w2 = Var(within=Binary)
        model.x2 = Var(within=Binary)
        model.yb = Var(within=Binary, bounds=(1,1))
        model.zb = Var(within=Binary, bounds=(0,0))
        model.yi = Var(within=Integers, bounds=(-1,None))
        model.zi = Var(within=Integers, bounds=(None,1))

        model.obj = Objective(expr=\
                                  model.w2 - model.x2 +\
                                  model.yb - model.zb +\
                                  model.yi - model.zi)

        model.c3 = Constraint(expr=model.w2 >= 0)
        model.c4 = Constraint(expr=model.x2 <= 1)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.w2.value = None
        model.x2.value = 1
        model.yb.value = 0
        model.zb.value = 1
        model.yi.value = None
        model.zi.value = 0

@register_model
class MILP_discrete_var_bounds_kernel(MILP_discrete_var_bounds):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.w2 = pmo.variable(domain=pmo.BooleanSet)
        model.x2 = pmo.variable(domain_type=pmo.IntegerSet,
                                lb=0, ub=1)
        model.yb = pmo.variable(domain_type=pmo.IntegerSet,
                                lb=1, ub=1)
        model.zb = pmo.variable(domain_type=pmo.IntegerSet,
                                lb=0, ub=0)
        model.yi = pmo.variable(domain=pmo.IntegerSet, lb=-1)
        model.zi = pmo.variable(domain=pmo.IntegerSet, ub=1)

        model.obj = pmo.objective(model.w2 - model.x2 +\
                                  model.yb - model.zb +\
                                  model.yi - model.zi)

        model.c3 = pmo.constraint(model.w2 >= 0)
        model.c4 = pmo.constraint(model.x2 <= 1)
