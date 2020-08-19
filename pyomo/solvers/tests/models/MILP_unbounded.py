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
from pyomo.core import ConcreteModel, Var, Objective, Integers
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model

@register_model
class MILP_unbounded(_BaseTestModel):
    """
    An unbounded mixed-integer linear program
    """

    description = "MILP_unbounded"
    capabilities = set(['linear', 'integer'])

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.solve_should_fail = True
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(within=Integers)
        model.y = Var(within=Integers)

        model.o = Objective(expr=model.x+model.y)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x.value = None
        model.y.value = None

    def post_solve_test_validation(self, tester, results):
        if tester is None:
            assert results['Solver'][0]['termination condition'] in \
                (TerminationCondition.unbounded,
                 TerminationCondition.infeasibleOrUnbounded)
        else:
            tester.assertIn(results['Solver'][0]['termination condition'],
                            (TerminationCondition.unbounded,
                             TerminationCondition.infeasibleOrUnbounded))

@register_model
class MILP_unbounded_kernel(MILP_unbounded):

    def _generate_model(self):
        self.model = pmo.block()
        model = self.model
        model._name = self.description

        model.x = pmo.variable(domain=pmo.IntegerSet)
        model.y = pmo.variable(domain=pmo.IntegerSet)

        model.o = pmo.objective(model.x+model.y)
