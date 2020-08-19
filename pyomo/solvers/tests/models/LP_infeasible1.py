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
from pyomo.core import ConcreteModel, Var, Objective, Constraint
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class LP_infeasible1(_BaseTestModel):
    """
    An infeasible LP
    """

    description = "LP_infeasible1"
    capabilities = set(['linear'])
    test_pickling = False

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.solve_should_fail = True
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(bounds=(1,None))
        model.y = Var(bounds=(1,None))
        model.o = Objective(expr=model.x+model.y)
        model.c = Constraint(expr=model.x+model.y <= 0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None

    def post_solve_test_validation(self, tester, results):
        if tester is None:
            assert results['Solver'][0]['termination condition'] == \
                TerminationCondition.infeasible
        else:
            tester.assertEqual(results['Solver'][0]['termination condition'],
                               TerminationCondition.infeasible)

@register_model
class LP_infeasible1_kernel(LP_infeasible1):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.x = pmo.variable(lb=1)
        model.y = pmo.variable(lb=1)
        model.o = pmo.objective(model.x+model.y)
        model.c = pmo.constraint(model.x+model.y <= 0)
