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
    DomainPartitioningMethod,
)
from pyomo.core.expr.compare import assertExpressionsStructurallyEqual
from pyomo.environ import ConcreteModel, Var, Constraint, TransformationFactory, log

## debug
from pytest import set_trace


class TestNonlinearToPWL_1D(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1, 10))
        m.cons = Constraint(expr=log(m.x) >= 0.35)

        return m

    def check_pw_linear_log_x(self, m, pwlf, x1, x2, x3):
        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')

        points = [(x1,), (x2,), (x3,)]
        self.assertEqual(pwlf._simplices, [(0, 1), (1, 2)])
        self.assertEqual(pwlf._points, points)
        self.assertEqual(len(pwlf._linear_functions), 2)

        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[0](m.x),
            ((log(x2) - log(x1)) / (x2 - x1)) * m.x
            + (log(x2) - ((log(x2) - log(x1)) / (x2 - x1)) * x2),
            places=7,
        )
        assertExpressionsStructurallyEqual(
            self,
            pwlf._linear_functions[1](m.x),
            ((log(x3) - log(x2)) / (x3 - x2)) * m.x
            + (log(x3) - ((log(x3) - log(x2)) / (x3 - x2)) * x3),
            places=7,
        )

        self.assertEqual(len(pwlf._expressions), 1)
        new_cons = n_to_pwl.get_transformed_component(m.cons)
        self.assertTrue(new_cons.active)
        self.assertIs(new_cons.body, pwlf._expressions[id(new_cons.body.expr)])
        self.assertIsNone(new_cons.ub)
        self.assertEqual(new_cons.lb, 0.35)
        self.assertIs(n_to_pwl.get_src_component(new_cons), m.cons)

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

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        points = [(1.0009,), (5.5,), (9.9991,)]
        (x1, x2, x3) = 1.0009, 5.5, 9.9991
        self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)

    def test_log_constraint_random_grid(self):
        m = self.make_model()

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        # [ESJ 3/30/24]: The seed is actually set in the function for getting
        # the points right now, so this will be deterministic.
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.RANDOM_GRID,
        )

        # cons is transformed
        self.assertFalse(m.cons.active)

        pwlf = list(
            m.component_data_objects(PiecewiseLinearFunction, descend_into=True)
        )
        self.assertEqual(len(pwlf), 1)
        pwlf = pwlf[0]

        x1 = 4.370861069626263
        x2 = 7.587945476302646
        x3 = 9.556428757689245
        self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)

    # def test_log_constraint_lmt_uniform_sample(self):
    #     m = self.make_model()

    #     n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
    #     n_to_pwl.apply_to(
    #         m,
    #         num_points=3,
    #         domain_partitioning_method=DomainPartitioningMethod.LINEAR_MODEL_TREE_UNIFORM,
    #     )

    #     # cons is transformed
    #     self.assertFalse(m.cons.active)

    #     pwlf = list(m.component_data_objects(PiecewiseLinearFunction,
    #                                          descend_into=True))
    #     self.assertEqual(len(pwlf), 1)
    #     pwlf = pwlf[0]

    #     set_trace()

    #     # TODO
    #     x1 = 4.370861069626263
    #     x2 = 7.587945476302646
    #     x3 = 9.556428757689245
    #     self.check_pw_linear_log_x(m, pwlf, x1, x2, x3)


class TestNonlinearToPWLIntegration(unittest.TestCase):
    def test_Ali_example(self):
        m = ConcreteModel()
        m.flow_super_heated_vapor = Var()
        m.flow_super_heated_vapor.fix(0.4586949988166174)
        m.super_heated_vapor_temperature = Var(bounds=(31, 200), initialize=45)
        m.evaporator_condensate_temperature = Var(
            bounds=(29, 120.8291392028045), initialize=30
        )
        m.LMTD = Var(bounds=(0, 130.61608989795093), initialize=1)
        m.evaporator_condensate_enthalpy = Var(
            bounds=(-15836.847, -15510.210751855624), initialize=100
        )
        m.evaporator_condensate_vapor_enthalpy = Var(
            bounds=(-13416.64, -13247.674383866839), initialize=100
        )
        m.heat_transfer_coef = Var(
            bounds=(1.9936854577372858, 5.995319594088982), initialize=0.1
        )
        m.evaporator_brine_temperature = Var(
            bounds=(27, 118.82913920280366), initialize=35
        )
        m.each_evaporator_area = Var()

        m.c = Constraint(
            expr=m.each_evaporator_area
            == (
                1.873
                * m.flow_super_heated_vapor
                * (
                    m.super_heated_vapor_temperature
                    - m.evaporator_condensate_temperature
                )
                / (100 * m.LMTD)
                + m.flow_super_heated_vapor
                * (
                    m.evaporator_condensate_vapor_enthalpy
                    - m.evaporator_condensate_enthalpy
                )
                / (
                    m.heat_transfer_coef
                    * (
                        m.evaporator_condensate_temperature
                        - m.evaporator_brine_temperature
                    )
                )
            )
        )

        n_to_pwl = TransformationFactory('contrib.piecewise.nonlinear_to_pwl')
        n_to_pwl.apply_to(
            m,
            num_points=3,
            domain_partitioning_method=DomainPartitioningMethod.UNIFORM_GRID,
        )

        m.pprint()

        from pyomo.environ import SolverFactory
        SolverFactory('gurobi').solve(m, tee=True)
