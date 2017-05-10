#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.core import ConcreteModel, Param, Var, Expression, Objective, Constraint, NonNegativeReals, maximize
from pyomo.opt import TerminationCondition
from pyomo.solvers.tests.models.base import _BaseTestModel, register_model


@register_model
class LP_infeasible2(_BaseTestModel):
    """
    An infeasible LP
    """

    description = "LP_infeasible2"
    capabilities = set(['linear'])
    test_pickling = False

    def __init__(self):
        _BaseTestModel.__init__(self)
        self.add_results(self.description+".json")

    def _generate_model(self):
        self.model = ConcreteModel()
        model = self.model
        model._name = self.description

        model.x = Var(bounds=(1,None))
        model.y = Var(bounds=(1,None))
        model.o = Objective(expr=-model.x-model.y, sense=maximize)
        model.c = Constraint(expr=model.x+model.y <= 0)

    def warmstart_model(self):
        assert self.model is not None
        model = self.model
        model.x = None
        model.y = None

    def post_solve_test_validation(self, tester, results):
        assert results['Solver'][0]['termination condition'] == TerminationCondition.infeasible

