# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common import unittest
from pyomo.common.numeric_types import value
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.compare import assertExpressionsStructurallyEqual
from pyomo.environ import (
    Block,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Reals,
    SolverFactory,
    TerminationCondition,
    TransformationFactory,
    Var,
    minimize,
)
from pyomo.opt import check_available_solvers

solvers = check_available_solvers('ipopt')


class TestKKT(unittest.TestCase):
    def check_primal_kkt_transformation_solns(self, m, m_reform):
        kkt = TransformationFactory('core.kkt')

        m.dual = Suffix(direction=Suffix.IMPORT)

        opt = SolverFactory('ipopt', options={"tol": 1e-8})
        results = opt.solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        results = opt.solve(m_reform)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        # equality constraint
        self.assertAlmostEqual(
            value(abs(kkt.get_multiplier_from_object(m_reform, m_reform.c2))),
            value(abs(m.dual[m.c2])),
            delta=1e-6,
        )

        # inequality constraints
        lower_bound_mult, upper_bound_mult = kkt.get_multiplier_from_object(
            m_reform, m_reform.c1
        )
        self.assertIsNone(lower_bound_mult)
        self.assertAlmostEqual(
            value(abs(upper_bound_mult)), value(abs(m.dual[m.c1])), delta=1e-6
        )
        lower_bound_mult, upper_bound_mult = kkt.get_multiplier_from_object(
            m_reform, m_reform.c3
        )
        self.assertAlmostEqual(
            value(abs(lower_bound_mult)), value(abs(m.dual[m.c3])), delta=1e-6
        )
        self.assertIsNone(upper_bound_mult)
        lower_bound_mult, upper_bound_mult = kkt.get_multiplier_from_object(
            m_reform, m_reform.c4
        )
        self.assertIsNone(lower_bound_mult)
        self.assertAlmostEqual(
            value(abs(upper_bound_mult)), value(abs(m.dual[m.c4])), delta=1e-6
        )

        for v in [(m.x, m_reform.x), (m.y, m_reform.y)]:
            primal_var, kkt_reform_var = v
            self.assertAlmostEqual(value(primal_var), value(kkt_reform_var))

    @unittest.skipIf('ipopt' not in solvers, "ipopt solver is not available")
    def test_kkt_solve(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=Reals)
        m.z = Var(domain=Reals)

        m.obj = Objective(
            expr=(m.x - 3) ** 2 + (m.y - 2) ** 2 + (m.z - 1) ** 2, sense=minimize
        )
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        m.c2 = Constraint(expr=m.x + m.y + m.z == 5)
        m.c3 = Constraint(expr=m.z >= 1)
        m.c4 = Constraint(expr=2 * m.x - m.y <= 4)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(m_reform)
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

    def test_kkt(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=Reals)
        m.z = Var(domain=Reals)

        m.obj = Objective(
            expr=(m.x - 3) ** 2 + (m.y - 2) ** 2 + (m.z - 1) ** 2, sense=minimize
        )
        # upper bounded constraint
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        # equality constraint
        m.c2 = Constraint(expr=m.x + m.y + m.z == 5)
        # lower bounded constraint
        m.c3 = Constraint(expr=m.z >= 1)
        # upper bounded constraint
        m.c4 = Constraint(expr=2 * m.x - m.y <= 4)

        kkt = TransformationFactory('core.kkt')
        kkt.apply_to(m)

        # equality constraint
        gamma0 = kkt.get_multiplier_from_object(m, m.c2)

        self.assertIs(kkt.get_object_from_multiplier(m, gamma0), m.c2)

        # upper bounded constraint
        alpha_con0_mults = kkt.get_multiplier_from_object(m, m.c1)
        alpha_con0_lower_mult = alpha_con0_mults[0]  # None
        alpha_con0_upper_mult = alpha_con0_mults[1]

        self.assertIsNone(alpha_con0_lower_mult)
        self.assertIs(kkt.get_object_from_multiplier(m, alpha_con0_upper_mult), m.c1)

        # lower bounded constraint
        alpha_con1_mults = kkt.get_multiplier_from_object(m, m.c3)
        alpha_con1_lower_mult = alpha_con1_mults[0]
        alpha_con1_upper_mult = alpha_con1_mults[1]  # None

        self.assertIs(kkt.get_object_from_multiplier(m, alpha_con1_lower_mult), m.c3)
        self.assertIsNone(alpha_con1_upper_mult)

        # upper bounded constraint
        alpha_con2_mults = kkt.get_multiplier_from_object(m, m.c4)
        alpha_con2_lower_mult = alpha_con2_mults[0]  # None
        alpha_con2_upper_mult = alpha_con2_mults[1]

        self.assertIsNone(alpha_con2_lower_mult)
        self.assertIs(kkt.get_object_from_multiplier(m, alpha_con2_upper_mult), m.c4)

        c2 = kkt.get_object_from_multiplier(m, gamma0)
        c1 = kkt.get_object_from_multiplier(m, alpha_con0_upper_mult)
        c3 = kkt.get_object_from_multiplier(m, alpha_con1_lower_mult)
        c4 = kkt.get_object_from_multiplier(m, alpha_con2_upper_mult)

        self.assertIs(kkt.get_multiplier_from_object(m, c2), gamma0)
        self.assertIs(kkt.get_multiplier_from_object(m, c1), alpha_con0_mults)
        self.assertIs(kkt.get_multiplier_from_object(m, c3), alpha_con1_mults)
        self.assertIs(kkt.get_multiplier_from_object(m, c4), alpha_con2_mults)

        self.assertIs(gamma0.ctype, Var)
        self.assertEqual(gamma0.domain, Reals)
        self.assertIsNone(gamma0.ub)
        self.assertIsNone(gamma0.lb)

        self.assertIs(alpha_con0_upper_mult.ctype, Var)
        self.assertEqual(alpha_con0_upper_mult.domain, NonNegativeReals)
        self.assertIsNone(alpha_con0_upper_mult.ub)

        self.assertIs(alpha_con1_lower_mult.ctype, Var)
        self.assertEqual(alpha_con1_lower_mult.domain, NonNegativeReals)
        self.assertIsNone(alpha_con1_lower_mult.ub)

        self.assertIs(alpha_con2_upper_mult.ctype, Var)
        self.assertEqual(alpha_con2_upper_mult.domain, NonNegativeReals)
        self.assertIsNone(alpha_con2_upper_mult.ub)

        self.assertIs(c1.ctype, Constraint)
        self.assertIs(c2.ctype, Constraint)
        self.assertIs(c3.ctype, Constraint)
        self.assertIs(c4.ctype, Constraint)

        # test Lagrangean expression
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.lagrangean.expr,
            (m.x - 3) ** 2
            + (m.y - 2) ** 2
            + (m.z - 1) ** 2
            + (m.x**2 + m.y**2 - 9) * alpha_con0_upper_mult
            + (5 - (m.x + m.y + m.z)) * gamma0
            + (1 - m.z) * alpha_con1_lower_mult
            + (2 * m.x - m.y - 4) * alpha_con2_upper_mult,
        )

        # test stationarity conditions
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[1].expr,
            2 * alpha_con2_upper_mult
            - gamma0
            + 2 * alpha_con0_upper_mult * m.x
            + 2 * (m.x - 3)
            == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[2].expr,
            -alpha_con2_upper_mult
            - gamma0
            + 2 * alpha_con0_upper_mult * m.y
            + 2 * (m.y - 2)
            == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[3].expr,
            -alpha_con1_lower_mult - gamma0 + 2 * (m.z - 1) == 0,
        )

        # test complementarity constraints
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[0], 0 <= alpha_con0_upper_mult
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[1], m.x**2 + m.y**2 - 9.0 <= 0
        )

        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[0], 0 <= alpha_con1_lower_mult
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[1], 1.0 - m.z <= 0
        )

        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[3]._args[0], 0 <= alpha_con2_upper_mult
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[3]._args[1], 2 * m.x - m.y - 4.0 <= 0
        )

        self.assertFalse(m.obj.active)

    def get_bilevel_model(self):
        m = ConcreteModel(name='bilevel')

        m.outer1 = Var(domain=Reals)
        m.outer2 = Var(domain=Reals)

        # Inner (follower) variables - decision variables for KKT conditions
        m.x = Var(domain=Reals)
        m.y = Var(domain=Reals)
        m.z = Var(domain=Reals)

        # Inner problem objective (depends on outer variables)
        m.obj = Objective(
            expr=(m.x - m.outer1) ** 2 + (m.y - 2) ** 2 + (m.z - m.outer2) ** 2,
            sense=minimize,
        )

        # Inner problem constraints (some depend on outer variables)
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9 + m.outer1)
        m.c2 = Constraint(expr=m.x + m.y + m.z == 5 + m.outer2)
        m.c3 = Constraint(expr=m.z >= 1)
        m.c4 = Constraint(expr=2 * m.x - m.y <= 4 + 0.5 * m.outer1)

        return m

    def test_parametrized_kkt(self):
        m = self.get_bilevel_model()

        kkt = TransformationFactory('core.kkt')
        kkt.apply_to(m, parameterize_wrt=[m.outer1, m.outer2])
        TransformationFactory("mpec.simple_nonlinear").apply_to(m)

        # equality constraint
        gamma0 = kkt.get_multiplier_from_object(m, m.c2)

        self.assertIs(kkt.get_object_from_multiplier(m, gamma0), m.c2)

        # upper bounded constraint
        alpha_con0_mults = kkt.get_multiplier_from_object(m, m.c1)
        alpha_con0_lower_mult = alpha_con0_mults[0]  # None
        alpha_con0_upper_mult = alpha_con0_mults[1]

        self.assertIsNone(alpha_con0_lower_mult)
        self.assertIs(kkt.get_object_from_multiplier(m, alpha_con0_upper_mult), m.c1)

        # lower bounded constraint
        alpha_con1_mults = kkt.get_multiplier_from_object(m, m.c3)
        alpha_con1_lower_mult = alpha_con1_mults[0]
        alpha_con1_upper_mult = alpha_con1_mults[1]  # None

        self.assertIs(kkt.get_object_from_multiplier(m, alpha_con1_lower_mult), m.c3)
        self.assertIsNone(alpha_con1_upper_mult)

        # upper bounded constraint
        alpha_con2_mults = kkt.get_multiplier_from_object(m, m.c4)
        alpha_con2_lower_mult = alpha_con2_mults[0]  # None
        alpha_con2_upper_mult = alpha_con2_mults[1]

        self.assertIsNone(alpha_con2_lower_mult)
        self.assertIs(kkt.get_object_from_multiplier(m, alpha_con2_upper_mult), m.c4)

        c2 = kkt.get_object_from_multiplier(m, gamma0)
        c1 = kkt.get_object_from_multiplier(m, alpha_con0_upper_mult)
        c3 = kkt.get_object_from_multiplier(m, alpha_con1_lower_mult)
        c4 = kkt.get_object_from_multiplier(m, alpha_con2_upper_mult)

        self.assertIs(kkt.get_multiplier_from_object(m, c2), gamma0)
        self.assertIs(kkt.get_multiplier_from_object(m, c1), alpha_con0_mults)
        self.assertIs(kkt.get_multiplier_from_object(m, c3), alpha_con1_mults)
        self.assertIs(kkt.get_multiplier_from_object(m, c4), alpha_con2_mults)

        self.assertIs(gamma0.ctype, Var)
        self.assertEqual(gamma0.domain, Reals)
        self.assertIsNone(gamma0.ub)
        self.assertIsNone(gamma0.lb)

        self.assertIs(alpha_con0_upper_mult.ctype, Var)
        self.assertEqual(alpha_con0_upper_mult.domain, NonNegativeReals)
        self.assertIsNone(alpha_con0_upper_mult.ub)

        self.assertIs(alpha_con1_lower_mult.ctype, Var)
        self.assertEqual(alpha_con1_lower_mult.domain, NonNegativeReals)
        self.assertIsNone(alpha_con1_lower_mult.ub)

        self.assertIs(alpha_con2_upper_mult.ctype, Var)
        self.assertEqual(alpha_con2_upper_mult.domain, NonNegativeReals)
        self.assertIsNone(alpha_con2_upper_mult.ub)

        self.assertIs(c1.ctype, Constraint)
        self.assertIs(c2.ctype, Constraint)
        self.assertIs(c3.ctype, Constraint)
        self.assertIs(c4.ctype, Constraint)

        # test Lagrangean expression
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.lagrangean.expr,
            (m.x - m.outer1) ** 2
            + (m.y - 2) ** 2
            + (m.z - m.outer2) ** 2
            + (m.x**2 + m.y**2 - (9 + m.outer1)) * alpha_con0_upper_mult
            + (-(m.x + m.y + m.z - (5 + m.outer2))) * gamma0
            + (1 - m.z) * alpha_con1_lower_mult
            + (2 * m.x - m.y - (4 + 0.5 * m.outer1)) * alpha_con2_upper_mult,
        )

        # test stationarity conditions
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[1].expr,
            (2 * alpha_con2_upper_mult - gamma0)
            + 2 * alpha_con0_upper_mult * m.x
            + 2 * (m.x - m.outer1)
            == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[2].expr,
            -alpha_con2_upper_mult
            - gamma0
            + 2 * alpha_con0_upper_mult * m.y
            + 2 * (m.y - 2)
            == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[3].expr,
            -alpha_con1_lower_mult - gamma0 + 2 * (m.z - m.outer2) == 0,
        )

        # test complementarity constraints
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[0], 0 <= alpha_con0_upper_mult
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[1], m.x**2 + m.y**2 - (9 + m.outer1) <= 0
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[0], 0 <= alpha_con1_lower_mult
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[1], 1 - m.z <= 0
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[3]._args[0], 0 <= alpha_con2_upper_mult
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.complements[3]._args[1],
            2 * m.x - m.y - (4 + 0.5 * m.outer1) <= 0,
        )

        self.assertFalse(m.obj.active)

    @unittest.skipIf('ipopt' not in solvers, "ipopt solver is not available")
    def test_solve_parametrized_kkt(self):
        m = self.get_bilevel_model()

        # test with a few values
        m.outer1.fix(1)
        m.outer2.fix(1)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(
            m_reform, parameterize_wrt=[m_reform.outer1, m_reform.outer2]
        )
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

        m.outer1.fix(1)
        m.outer2.fix(5)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(
            m_reform, parameterize_wrt=[m_reform.outer1, m_reform.outer2]
        )
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

        m.outer1.fix(3)
        m.outer2.fix(3)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(
            m_reform, parameterize_wrt=[m_reform.outer1, m_reform.outer2]
        )
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

    def test_multiple_obj_error(self):
        m = self.get_bilevel_model()
        m.obj.deactivate()
        kkt = TransformationFactory('core.kkt')

        with self.assertRaisesRegex(
            ValueError, "model must have exactly one active objective; found 0"
        ):
            kkt.apply_to(m)

    def test_kkt_block_name_error(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=Reals)
        m.obj = Objective(expr=(m.x - 3) ** 2, sense=minimize)
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        m.b1 = Block()
        kkt = TransformationFactory('core.kkt')

        with self.assertRaisesRegex(
            ValueError,
            "model already has an attribute with the " "specified kkt_block_name: 'b1'",
        ):
            kkt.apply_to(m, kkt_block_name='b1')

    def test_parameterize_wrt_unknown_error(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=Reals)
        m.obj = Objective(expr=(m.x - 3) ** 2, sense=minimize)
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        m.b1 = Block()
        m.b1.x1 = Var(domain=Reals)
        m.b1.deactivate()
        kkt = TransformationFactory('core.kkt')

        with self.assertRaisesRegex(
            ValueError,
            "A variable passed in parameterize_wrt does not exist in an "
            "active constraint or objective within the model. "
            "Invalid variables:\n\t" + "b1.x1",
        ):
            kkt.apply_to(m, parameterize_wrt=[m.b1.x1])

    def test_get_object_from_multiplier_error(self):
        m = ConcreteModel(name="model")
        m.x = Var(domain=Reals)
        m.y = Var(domain=Reals)
        m.obj = Objective(expr=(m.x - 3) ** 2, sense=minimize)
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        kkt = TransformationFactory('core.kkt')
        kkt.apply_to(m)

        m2 = ConcreteModel()
        m2.gamma = Var(domain=Reals)

        with self.assertRaisesRegex(
            ValueError, f"The KKT multiplier: {m2.gamma}, does not exist on model."
        ):
            kkt.get_object_from_multiplier(m, m2.gamma)

    def test_get_multiplier_from_object_error(self):
        m = ConcreteModel(name="model")
        m.x = Var(domain=Reals, bounds=(0, 10))
        m.y = Var(domain=Reals)
        m.obj = Objective(expr=(m.x - 3) ** 2, sense=minimize)
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        m.c2 = Constraint(expr=(0, m.y, 10))
        kkt = TransformationFactory('core.kkt')
        kkt.apply_to(m)

        m2 = ConcreteModel()
        m2.new_con = Constraint(expr=m.x <= 5)

        with self.assertRaisesRegex(
            ValueError,
            "The component 'new_con' either does not exist on 'model', "
            "or is not associated with a multiplier.",
        ):
            kkt.get_multiplier_from_object(m, component=m2.new_con)
