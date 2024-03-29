#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.contrib.piecewise import PiecewiseLinearFunction
from pyomo.contrib.piecewise.transform.nonlinear_to_pwl import (
    NonlinearToPWL,
    DomainPartitioningMethod
)
from pyomo.core.expr.compare import (
    assertExpressionsStructurallyEqual,
)
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    TransformationFactory,
    log,
)

## debug
from pytest import set_trace

class TestNonlinearToPWL_1D(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.cons = Constraint(expr=log(m.x) >= 0.35)

        return m
        
    def test_log_constraint_uniform_grid(self):
        m = self.make_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        # cons is transformed
        self.assertFalse(m.cons.active)

        pwlf = list(m.component_data_objects(PiecewiseLinearFunction,
                                             descend_into=True))
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]
        
        points = [(1.0009,), (5.5,), (9.9991,)]
        self.assertEqual(pwlf._simplices, [(0, 1), (1, 2)])
        self.assertEqual(pwlf._points, points)
        self.assertEqual(len(pwlf._linear_functions), 2)

        x1 = 1.0009
        x2 = 5.5
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[0](m.x),
            ((log(x2) - log(x1))/(x2 - x1))*m.x +
            (log(x2) - ((log(x2) - log(x1))/(x2 - x1))*x2),
            places=7
        )
        x1 = 5.5
        x2 = 9.9991
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[1](m.x),
            ((log(x2) - log(x1))/(x2 - x1))*m.x +
            (log(x2) - ((log(x2) - log(x1))/(x2 - x1))*x2),
            places=7
        )

        self.assertEqual(len(pwlf._expressions), 1)
        new_cons = n_to_pwl.get_transformed_component(m.cons)
        self.assertTrue(new_cons.active)
        self.assertIs(new_cons.body, pwlf._expressions[id(new_cons.body.expr)])
        self.assertIsNone(new_cons.ub)
        self.assertEqual(new_cons.lb, 0.35)
        self.assertIs(n_to_pwl.get_src_component(new_cons), m.cons)
