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
from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class LP_simple(_BaseTestModel):
    """
    A simple linear program
    """

    description = "LP_simple"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.a1 = Param(initialize=1.0, mutable=True)
        model.a2 = Param([1], initialize=1.0, mutable=True)
        model.a3 = Param(initialize=1.0, mutable=False)
        model.a4 = Param([1], initialize=1.0, mutable=False)
        model.x = Var(within=NonNegativeReals)
        model.y = Var(within=NonNegativeReals)
        model.z1 = Var(bounds=(float('-inf'), float('inf')))
        model.z2 = Var()
        model.dummy_expr1 = Expression(initialize=model.a1*model.a2[1])
        model.dummy_expr2 = Expression(initialize=model.y/model.a3*model.a4[1])

        model.inactive_obj = Objective(
            expr=model.x + 3.0*model.y + 1.0 + model.z1 - model.z2)
        model.inactive_obj.deactivate()
        model.p = Param(mutable=True, initialize=0.0)
        model.obj = Objective(expr=model.p + model.inactive_obj)

        model.c1 = Constraint(expr=model.dummy_expr1 <= model.dummy_expr2)
        model.c2 = Constraint(expr=(2.0, model.x/model.a3 - model.y, 10))
        model.c3 = Constraint(expr=(0, model.z1 + 1, 10))
        model.c4 = Constraint(expr=(-10, model.z2 + 1, 0))

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = 1.0

@register_model
class LP_simple_kernel(LP_simple):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.a1 = pmo.parameter(value=1.0)
        model.a2 = pmo.parameter_dict(
            {1: pmo.parameter(value=1.0)})
        model.a3 = pmo.parameter(value=1.0)
        model.a4 = pmo.parameter_dict(
            {1: pmo.parameter(value=1.0)})
        model.x = pmo.variable(domain=NonNegativeReals)
        model.y = pmo.variable(domain=NonNegativeReals)
        model.z1 = pmo.variable()
        model.z2 = pmo.variable()
        model.dummy_expr1 = pmo.expression(model.a1*model.a2[1])
        model.dummy_expr2 = pmo.expression(model.y/model.a3*model.a4[1])

        model.inactive_obj = pmo.objective(
            model.x + 3.0*model.y + 1.0 + model.z1 - model.z2)
        model.inactive_obj.deactivate()
        model.p = pmo.parameter(value=0.0)
        model.obj = pmo.objective(model.p + model.inactive_obj)

        model.c1 = pmo.constraint(model.dummy_expr1 <= pmo.noclone(model.dummy_expr2))
        model.c2 = pmo.constraint((2.0, model.x/model.a3 - model.y, 10))
        model.c3 = pmo.constraint((0, model.z1 + 1, 10))
        model.c4 = pmo.constraint((-10, model.z2 + 1, 0))
