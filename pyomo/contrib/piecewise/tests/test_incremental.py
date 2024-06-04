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
from pyomo.contrib.piecewise.triangulations import Triangulation
from pyomo.core.base import TransformationFactory
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import Constraint, SolverFactory, Var, ConcreteModel, Objective, log, value, maximize
from pyomo.contrib.piecewise import PiecewiseLinearFunction

from pyomo.contrib.piecewise.transform.incremental import IncrementalGDPTransformation

class TestTransformPiecewiseModelToIncrementalMIP(unittest.TestCase):

    def test_solve_log_model(self):
        m = make_log_x_model_ordered()
        TransformationFactory(
            'contrib.piecewise.incremental'
        ).apply_to(m)
        TransformationFactory(
            'gdp.bigm'
        ).apply_to(m)
        SolverFactory('gurobi').solve(m)
        ct.check_log_x_model_soln(self, m)
    
    #def test_solve_univariate_log_model(self):
    #    m = ConcreteModel()
    #    m.x = Var(bounds=(1, 10))
    #    m.pw_log = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=log)

    #    # Here are the linear functions, for safe keeping.
    #    def f1(x):
    #        return (log(3) / 2) * x - log(3) / 2

    #    m.f1 = f1

    #    def f2(x):
    #        return (log(2) / 3) * x + log(3 / 2)

    #    m.f2 = f2

    #    def f3(x):
    #        return (log(5 / 3) / 4) * x + log(6 / ((5 / 3) ** (3 / 2)))

    #    m.f3 = f3

    #    m.log_expr = m.pw_log(m.x)
    #    m.obj = Objective(expr=m.log_expr, sense=maximize)

    #    TransformationFactory(
    #        'contrib.piecewise.incremental'
    #    ).apply_to(m)
    #    m.pprint()
    #    TransformationFactory(
    #        'gdp.bigm'
    #    ).apply_to(m)
    #    print('####### PPRINTNG AGAIN AFTER BIGM #######')
    #    m.pprint()
    #    # log is increasing so the optimal value should be log(10)
    #    SolverFactory('gurobi').solve(m)
    #    print(f"optimal value is {value(m.obj)}")
    #    self.assertTrue(abs(value(m.obj) - log(10)) < 0.001)


# Make a version of the log_x model with the simplices properly ordered for the
# incremental transform
def make_log_x_model_ordered():
    m = ConcreteModel()
    m.x = Var(bounds=(1, 10))
    m.pw_log = PiecewiseLinearFunction(points=[1, 3, 6, 10], function=log)
    m.pw_log._triangulation = Triangulation.AssumeValid

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
    m.obj = Objective(expr=m.log_expr)

    m.x1 = Var(bounds=(0, 3))
    m.x2 = Var(bounds=(1, 7))

    ## apprximates paraboloid x1**2 + x2**2
    def g1(x1, x2):
        return 3 * x1 + 5 * x2 - 4

    m.g1 = g1

    def g2(x1, x2):
        return 3 * x1 + 11 * x2 - 28

    m.g2 = g2
    # order for incremental transformation
    simplices = [
        [(0, 1), (3, 1), (3, 4)],
        [(3, 4), (0, 1), (0, 4)],
        [(0, 4), (0, 7), (3, 4)],
        [(3, 4), (3, 7), (0, 7)],
    ]
    m.pw_paraboloid = PiecewiseLinearFunction(
        simplices=simplices, linear_functions=[g1, g1, g2, g2]
    )
    m.pw_paraboloid._triangulation = Triangulation.AssumeValid
    m.paraboloid_expr = m.pw_paraboloid(m.x1, m.x2)

    def c_rule(m, i):
        if i == 0:
            return m.x >= m.paraboloid_expr
        else:
            return (1, m.x1, 2)

    m.indexed_c = Constraint([0, 1], rule=c_rule)

    return m