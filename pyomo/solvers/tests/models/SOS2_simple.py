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
from pyomo.core import ConcreteModel, Param, Var, Objective, Constraint, SOSConstraint, NonNegativeReals, ConstraintList, sum_product
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
        model.c1.add((0.0, model.lmbda[1], 1.0))
        model.c1.add((0.0, model.lmbda[2], 1.0))
        model.c1.add(0.0 <= model.lmbda[3])
        model.c2 = SOSConstraint(var=model.lmbda, index=[1,2,3], sos=2)
        model.c3 = Constraint(expr=sum_product(model.lmbda) == 1)
        model.c4 = Constraint(expr=model.f==sum_product(model.fi,model.lmbda))
        model.c5 = Constraint(expr=model.x==sum_product(model.xi,model.lmbda))
        model.x = 2.75
        model.x.fixed = True

        # Make an empty SOSConstraint
        model.c6 = SOSConstraint(var=model.lmbda, index=[1,2,3], sos=2)
        model.c6.set_items([],[])
        assert len(list(model.c6.get_items())) == 0

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.f.value = 0
        assert model.x.value == 2.75 # Fixed
        model.p.value = 1
        model.n.value = 0
        model.lmbda[1].value = None
        model.lmbda[2].value = None
        model.lmbda[3].value = 1

@register_model
class SOS2_simple_kernel(SOS2_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.f = pmo.variable()
        model.x = pmo.variable(lb=1,ub=3)
        model.fi = pmo.parameter_dict()
        model.fi[1] = pmo.parameter(value=1.0)
        model.fi[2] = pmo.parameter(value=2.0)
        model.fi[3] = pmo.parameter(value=0.0)
        model.xi = pmo.parameter_dict()
        model.xi[1] = pmo.parameter(value=1.0)
        model.xi[2] = pmo.parameter(value=2.0)
        model.xi[3] = pmo.parameter(value=3.0)
        model.p = pmo.variable(domain=NonNegativeReals)
        model.n = pmo.variable(domain=NonNegativeReals)
        model.lmbda = pmo.variable_dict(
            (i, pmo.variable()) for i in range(1,4))
        model.obj = pmo.objective(model.p+model.n)
        model.c1 = pmo.constraint_dict()
        model.c1[1] = pmo.constraint((0.0, model.lmbda[1], 1.0))
        model.c1[2] = pmo.constraint((0.0, model.lmbda[2], 1.0))
        model.c1[3] = pmo.constraint(0.0 <= model.lmbda[3])
        model.c2 = pmo.sos2(model.lmbda.values())
        model.c3 = pmo.constraint(sum(model.lmbda.values()) == 1)
        model.c4 = pmo.constraint(model.f==sum(model.fi[i]*model.lmbda[i]
                                               for i in model.lmbda))
        model.c5 = pmo.constraint(model.x==sum(model.xi[i]*model.lmbda[i]
                                               for i in model.lmbda))
        model.x.fix(2.75)

        # Make an empty SOS constraint
        model.c6 = pmo.sos2([])
