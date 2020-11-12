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
from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class LP_constant_objective2(_BaseTestModel):
    """
    A continuous linear model with a constant objective that
    starts as a linear expression
    """

    description = "LP_constant_objective2"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x-model.x)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 1.0

@register_model
class LP_constant_objective2_kernel(LP_constant_objective2):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.x = pmo.variable(domain=NonNegativeReals)
        model.obj = pmo.objective(model.x-model.x)
        model.con = pmo.constraint(model.x == 1.0)
