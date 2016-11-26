#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, Block, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class LP_block(_BaseTestModel):
    """
    A continuous linear model with nested blocks
    """

    description = "LP_block"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.b = Block()
        model.B = Block([1,2,3])
        model.a = Param(initialize=1.0, mutable=True)
        model.b.x = Var(within=NonNegativeReals)
        model.B[1].x = Var(within=NonNegativeReals)

        model.obj = Objective(expr=model.b.x + 3.0*model.B[1].x)
        model.obj.deactivate()
        model.B[2].c = Constraint(expr=-model.B[1].x <= -model.a)
        model.B[2].obj = Objective(expr=model.b.x + 3.0*model.B[1].x + 2)
        model.B[3].c = Constraint(expr=2.0 <= model.b.x/model.a - model.B[1].x <= 10)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.b.x = 1.0
        model.B[1].x = 1.0

