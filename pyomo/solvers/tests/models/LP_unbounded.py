#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.kernel as pmo
from pyomo.core import ConcreteModel, Var, Objective
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
        self.solve_should_fail = True
        self.add_results(self.description + ".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var()
        model.y = Var()

        model.o = Objective(expr=model.x + model.y)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None

    def post_solve_test_validation(self, tester, results):
        outcomes = [
            TerminationCondition.unbounded,
            TerminationCondition.infeasibleOrUnbounded,
        ]
        if '_gams_' in str(tester):
            # GAMS maps CPLEX's InfeasibleOrUnbounded to Infeasible
            outcomes.append(TerminationCondition.infeasible)
        if tester is None:
            assert results['Solver'][0]['termination condition'] in outcomes
        else:
            tester.assertIn(results['Solver'][0]['termination condition'], outcomes)


@register_model
class LP_unbounded_kernel(LP_unbounded):
    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.x = pmo.variable()
        model.y = pmo.variable()

        model.o = pmo.objective(model.x + model.y)
