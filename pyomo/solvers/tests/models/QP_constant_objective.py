#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

# NOTE: We could test this problem on solvers that only handle
#       linear objectives IF we could get some proper preprocessing
#       in place for the canonical_repn

@register_model
class QP_constant_objective(_BaseTestModel):
    """
    A continuous linear model with a constant objective that starts
    as quadratic expression
    """

    description = "QP_constant_objective"
    capabilities = set(['linear', 'quadratic_objective'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x**2-model.x**2)
        model.con = Constraint(expr=model.x == 1.0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = 1.0

@register_model
class QP_constant_objective_kernel(QP_constant_objective):

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=NonNegativeReals)
        model.obj = Objective(expr=model.x**2-model.x**2)
        model.con = Constraint(expr=model.x == 1.0)
