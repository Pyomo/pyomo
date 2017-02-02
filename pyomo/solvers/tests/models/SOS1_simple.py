#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals, SOSConstraint, summation
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class SOS1_simple(_BaseTestModel):
    """
    A simple linear program
    """

    description = "SOS1_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'integer', 'sos1'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.a = Param(initialize=0.1)
        model.x = Var(within=NonNegativeReals)
        model.y = Var([1,2],within=NonNegativeReals)

        model.obj = Objective(expr=model.x + model.y[1]+2*model.y[2])
        model.c1 = Constraint(expr=model.a <= model.y[2])
        model.c2 = Constraint(expr=2.0 <= model.x <= 10.0)
        model.c3 = SOSConstraint(var=model.y, index=[1,2], sos=1)
        model.c4 = Constraint(expr=summation(model.y) == 1)

        # Make an empty SOSConstraint
        model.c5 = SOSConstraint(var=model.y, index=[1,2], sos=1)
        model.c5.set_items([],[])
        assert len(list(model.c5.get_items())) == 0

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = 0
        model.y[1] = 1
        model.y[2] = None

