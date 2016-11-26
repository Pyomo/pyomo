#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, Binary, maximize
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class MIQCP_simple(_BaseTestModel):
    """
    A mixed-integer model with a quadratic objective and quadratic constraints
    """

    description = "MIQCP_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'integer', 'quadratic_constraint'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=Binary)
        model.y = Var(within=Binary)
        model.z = Var(within=Binary)

        model.obj = Objective(expr=model.x,sense=maximize)
        model.c0 = Constraint(expr=model.x+model.y+model.z == 1)
        model.qc0 = Constraint(expr=model.x**2 + model.y**2 <= model.z**2)
        model.qc1 = Constraint(expr=model.x**2 <= model.y*model.z)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = None
        model.z = None

