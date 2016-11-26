#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, SOSConstraint, NonNegativeReals, ConstraintList, summation
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class SOS2_simple(_BaseTestModel):
    """
    A discrete linear model with sos2 constraints
    """

    description = "SOS2_simple"
    level = ('nightly', 'expensive')
    capabilities = set(['linear', 'integer', 'sos2'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.f = Var()
        model.x = Var(bounds=(1,3))
        model.fi = Param([1,2,3],mutable=True)
        model.fi[1] = 1.0
        model.fi[2] = 2.0
        model.fi[3] = 0.0
        model.xi = Param([1,2,3],mutable=True)
        model.xi[1] = 1.0
        model.xi[2] = 2.0
        model.xi[3] = 3.0
        model.p = Var(within=NonNegativeReals)
        model.n = Var(within=NonNegativeReals)
        model.lmbda = Var([1,2,3])
        model.obj = Objective(expr=model.p+model.n)
        model.c1 = ConstraintList()
        model.c1.add(0.0 <= model.lmbda[1] <= 1.0)
        model.c1.add(0.0 <= model.lmbda[2] <= 1.0)
        model.c1.add(0.0 <= model.lmbda[3])
        model.c2 = SOSConstraint(var=model.lmbda, index=[1,2,3], sos=2)
        model.c3 = Constraint(expr=summation(model.lmbda) == 1)
        model.c4 = Constraint(expr=model.f==summation(model.fi,model.lmbda))
        model.c5 = Constraint(expr=model.x==summation(model.xi,model.lmbda))
        model.x = 2.75
        model.x.fixed = True

        # Make an empty SOSConstraint
        model.c6 = SOSConstraint(var=model.lmbda, index=[1,2,3], sos=2)
        model.c6.set_items([],[])
        assert len(list(model.c6.get_items())) == 0

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.f = 0
        model.x = 2.75 # Fixed
        model.p = 1
        model.n = 0
        model.lmbda[1] = None
        model.lmbda[2] = None
        model.lmbda[3] = 1

