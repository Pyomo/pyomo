#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, Binary, Integers, NonNegativeReals
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
        model.w2 = None
        model.x2 = 1
        model.yb = 0
        model.zb = 1
        model.yi = None
        model.zi = 0

