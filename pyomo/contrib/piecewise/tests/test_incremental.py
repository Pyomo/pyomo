#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise.tests import models
import pyomo.contrib.piecewise.tests.common_tests as ct
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var, ConcreteModel, Objective, log, value, maximize
from pyomo.contrib.piecewise import PiecewiseLinearFunction

from pyomo.contrib.piecewise.transform.incremental import IncrementalInnerGDPTransformation
from pyomo.contrib.piecewise.transform.disagreggated_logarithmic import (
    DisaggregatedLogarithmicInnerGDPTransformation
)

class TestTransformPiecewiseModelToNestedInnerRepnGDP(unittest.TestCase):

    #def test_solve_log_model(self):
    #    m = models.make_log_x_model()
    #    TransformationFactory(
    #        'contrib.piecewise.incremental'
    #    ).apply_to(m)
    #    TransformationFactory(
    #        'gdp.bigm'
    #    ).apply_to(m)
    #    SolverFactory('gurobi').solve(m)
    #    ct.check_log_x_model_soln(self, m)
    
    def test_solve_univariate_log_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.pw_log = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=log)

        # Here are the linear functions, for safe keeping.
        def f1(x):
            return (log(3) / 2) * x - log(3) / 2

        m.f1 = f1

        def f2(x):
            return (log(2) / 3) * x + log(3 / 2)

        m.f2 = f2

        def f3(x):
            return (log(5 / 3) / 4) * x + log(6 / ((5 / 3) ** (3 / 2)))

        m.f3 = f3

        m.log_expr = m.pw_log(m.x)
        m.obj = Objective(expr=m.log_expr, sense=maximize)

        TransformationFactory(
            'contrib.piecewise.incremental'
            #'contrib.piecewise.disaggregated_logarithmic'
        ).apply_to(m)
        m.pprint()
        TransformationFactory(
            'gdp.bigm'
        ).apply_to(m)
        print('####### PPRINTNG AGAIN AFTER BIGM #######')
        m.pprint()
        # log is increasing so the optimal value should be log(10)
        SolverFactory('gurobi').solve(m)
        print(f"optimal value is {value(m.obj)}")
        self.assertTrue(abs(value(m.obj) - log(10)) < 0.001)