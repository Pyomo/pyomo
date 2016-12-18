#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class LP_unbounded(_BaseTestModel):
    """
    A unbounded linear program
    """

    description = "LP_unbounded"
    capabilities = set(['linear'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var()
        model.y = Var()

        model.o = Objective(expr=model.x+model.y)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = None

    def post_solve_test_validation(self, tester, results):
        assert results['Solver'][0]['termination condition'] == TerminationCondition.unbounded
