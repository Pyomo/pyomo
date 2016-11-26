#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals, Binary
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class MIQP_simple(_BaseTestModel):
    """
    A mixed-integer model with a quadratic objective and linear constraints
    """

    description = "MIQP_simple"

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.capabilities.add('linear')
        self.capabilities.add('integer')
        self.capabilities.add('quadratic_objective')
        self.level = ('nightly', 'expensive')
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.a = Param(initialize=1.0)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=Binary)

        model.obj = Objective(expr=model.x**2 + 3.0*model.y**2)
        model.c1 = Constraint(expr=model.a <= model.y)
        model.c2 = Constraint(expr=2.0 <= model.x/model.a - model.y <= 10)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = 1
        model.y = 1

