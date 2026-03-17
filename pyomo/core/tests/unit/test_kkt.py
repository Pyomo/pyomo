# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from pyomo.common.dependencies import scipy_available
from pyomo.common.numeric_types import value
import pyomo.common.unittest as unittest
from pyomo.common.autoslots import AutoSlots
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.environ import (
    ConcreteModel,
    Reals,
    Block,
    Constraint,
    ConstraintList,
    Expression,
    NonNegativeReals,
    Objective,
    RangeSet,
    Reals,
    Set,
    TransformationFactory,
    Var,
    maximize,
    minimize,
    SolverFactory,
    TerminationCondition,
)
from pyomo.core.base.suffix import Suffix
from pyomo.core.expr.calculus.diff_with_pyomo import reverse_sd
from pyomo.mpec import ComplementarityList, complements
from pyomo.util.vars_from_expressions import get_vars_from_components
from pyomo.util.config_domains import ComponentDataSet
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)


@unittest.skipUnless(scipy_available, "Scipy not available")
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

        for cons in [
            (m.c1, m_reform.c1),
            (m.c2, m_reform.c2),
            (m.c3, m_reform.c3),
            (m.c4, m_reform.c4),
        ]:
            primal_con, kkt_reform_con = cons
            self.assertAlmostEqual(
                value(
                    abs(kkt.get_multiplier_from_constraint(m_reform, kkt_reform_con))
                ),
                value(abs(m.dual[primal_con])),
                delta=1e-6,
            )

        for v in [(m.x, m_reform.x), (m.y, m_reform.y)]:
            primal_var, kkt_reform_var = v
            self.assertAlmostEqual(value(primal_var), value(kkt_reform_var))

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
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        m.c2 = Constraint(expr=m.x + m.y + m.z == 5)
        m.c3 = Constraint(expr=m.z >= 1)
        m.c4 = Constraint(expr=2 * m.x - m.y <= 4)

        kkt = TransformationFactory('core.kkt')
        kkt.apply_to(m)

        gamma0 = kkt.get_multiplier_from_constraint(m, m.c2)
        alpha_con0 = kkt.get_multiplier_from_constraint(m, m.c1)
        alpha_con1 = kkt.get_multiplier_from_constraint(m, m.c3)
        alpha_con2 = kkt.get_multiplier_from_constraint(m, m.c4)

        self.assertIs(kkt.get_constraint_from_multiplier(m, gamma0), m.c2)
        self.assertIs(kkt.get_constraint_from_multiplier(m, alpha_con0), m.c1)
        self.assertIs(kkt.get_constraint_from_multiplier(m, alpha_con1), m.c3)
        self.assertIs(kkt.get_constraint_from_multiplier(m, alpha_con2), m.c4)

        c2 = kkt.get_constraint_from_multiplier(m, gamma0)
        c1 = kkt.get_constraint_from_multiplier(m, alpha_con0)
        c3 = kkt.get_constraint_from_multiplier(m, alpha_con1)
        c4 = kkt.get_constraint_from_multiplier(m, alpha_con2)

        self.assertIs(kkt.get_multiplier_from_constraint(m, c2), gamma0)
        self.assertIs(kkt.get_multiplier_from_constraint(m, c1), alpha_con0)
        self.assertIs(kkt.get_multiplier_from_constraint(m, c3), alpha_con1)
        self.assertIs(kkt.get_multiplier_from_constraint(m, c4), alpha_con2)

        self.assertIs(gamma0.ctype, Var)
        self.assertEqual(gamma0.domain, Reals)
        self.assertIsNone(gamma0.ub)
        self.assertIsNone(gamma0.lb)
        self.assertIs(alpha_con0.ctype, Var)
        self.assertEqual(alpha_con0.domain, NonNegativeReals)
        self.assertIsNone(alpha_con0.ub)
        self.assertEqual(alpha_con1.lb, 0)
        self.assertIs(alpha_con1.ctype, Var)
        self.assertEqual(alpha_con1.domain, NonNegativeReals)
        self.assertIsNone(alpha_con1.ub)
        self.assertEqual(alpha_con2.lb, 0)
        self.assertIs(alpha_con2.ctype, Var)
        self.assertEqual(alpha_con2.domain, NonNegativeReals)
        self.assertIsNone(alpha_con2.ub)
        self.assertEqual(alpha_con2.lb, 0)

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
            + (5.0 - (m.x + m.y + m.z)) * gamma0
            + (m.x**2 + m.y**2 - 9.0) * alpha_con0
            + (1.0 - m.z) * alpha_con1
            + (2 * m.x - m.y - 4.0) * alpha_con2,
        )

        # test stationarity conditions
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[1].expr,
            2 * alpha_con2 + 2 * alpha_con0 * m.x - gamma0 + 2 * (m.x - 3) == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[2].expr,
            -alpha_con2 + 2 * alpha_con0 * m.y - gamma0 + 2 * (m.y - 2) == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[3].expr,
            -alpha_con1 - gamma0 + 2 * (m.z - 1) == 0,
        )

        # test complementarity constraints
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[0], 0 <= alpha_con0
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[1], m.x**2 + m.y**2 - 9.0 <= 0
        )

        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[0], 0 <= alpha_con1
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[1], 1.0 - m.z <= 0
        )

        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[3]._args[0], 0 <= alpha_con2
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[3]._args[1], 2 * m.x - m.y - 4.0 <= 0
        )

        self.assertFalse(m.obj.active)

        self.assertTrue(m.kkt.dummy_obj.active)
        self.assertEqual(m.kkt.dummy_obj.expr, 1.0)

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
        kkt.apply_to(m, parametrize_wrt=[m.outer1, m.outer2])
        TransformationFactory("mpec.simple_nonlinear").apply_to(m)

        gamma0 = kkt.get_multiplier_from_constraint(m, m.c2)
        alpha_con0 = kkt.get_multiplier_from_constraint(m, m.c1)
        alpha_con1 = kkt.get_multiplier_from_constraint(m, m.c3)
        alpha_con2 = kkt.get_multiplier_from_constraint(m, m.c4)

        self.assertIs(kkt.get_constraint_from_multiplier(m, gamma0), m.c2)
        self.assertIs(kkt.get_constraint_from_multiplier(m, alpha_con0), m.c1)
        self.assertIs(kkt.get_constraint_from_multiplier(m, alpha_con1), m.c3)
        self.assertIs(kkt.get_constraint_from_multiplier(m, alpha_con2), m.c4)

        c2 = kkt.get_constraint_from_multiplier(m, gamma0)
        c1 = kkt.get_constraint_from_multiplier(m, alpha_con0)
        c3 = kkt.get_constraint_from_multiplier(m, alpha_con1)
        c4 = kkt.get_constraint_from_multiplier(m, alpha_con2)

        self.assertIs(kkt.get_multiplier_from_constraint(m, c2), gamma0)
        self.assertIs(kkt.get_multiplier_from_constraint(m, c1), alpha_con0)
        self.assertIs(kkt.get_multiplier_from_constraint(m, c3), alpha_con1)
        self.assertIs(kkt.get_multiplier_from_constraint(m, c4), alpha_con2)

        self.assertIs(gamma0.ctype, Var)
        self.assertEqual(gamma0.domain, Reals)
        self.assertIsNone(gamma0.ub)
        self.assertIsNone(gamma0.lb)
        self.assertIs(alpha_con0.ctype, Var)
        self.assertEqual(alpha_con0.domain, NonNegativeReals)
        self.assertIsNone(alpha_con0.ub)
        self.assertEqual(alpha_con1.lb, 0)
        self.assertIs(alpha_con1.ctype, Var)
        self.assertEqual(alpha_con1.domain, NonNegativeReals)
        self.assertIsNone(alpha_con1.ub)
        self.assertEqual(alpha_con2.lb, 0)
        self.assertIs(alpha_con2.ctype, Var)
        self.assertEqual(alpha_con2.domain, NonNegativeReals)
        self.assertIsNone(alpha_con2.ub)
        self.assertEqual(alpha_con2.lb, 0)

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
            + (-(m.x + m.y + m.z - (5 + m.outer2))) * gamma0
            + (m.x**2 + m.y**2 - (9 + m.outer1)) * alpha_con0
            + (1.0 - m.z) * alpha_con1
            + (2 * m.x - m.y - (4 + 0.5 * m.outer1)) * alpha_con2,
        )

        # test stationarity conditions
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[1].expr,
            2 * alpha_con2 + 2 * alpha_con0 * m.x - gamma0 + 2 * (m.x - m.outer1) == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[2].expr,
            -alpha_con2 + 2 * alpha_con0 * m.y - gamma0 + 2 * (m.y - 2) == 0,
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.stationarity_conditions[3].expr,
            -alpha_con1 - gamma0 + 2 * (m.z - m.outer2) == 0,
        )

        # test complementarity constraints
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[0], 0 <= alpha_con0
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[1]._args[1], m.x**2 + m.y**2 - (9 + m.outer1) <= 0
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[0], 0 <= alpha_con1
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[2]._args[1], 1.0 - m.z <= 0
        )
        assertExpressionsStructurallyEqual(
            self, m.kkt.complements[3]._args[0], 0 <= alpha_con2
        )
        assertExpressionsStructurallyEqual(
            self,
            m.kkt.complements[3]._args[1],
            2 * m.x - m.y - (4 + 0.5 * m.outer1) <= 0,
        )

        self.assertFalse(m.obj.active)

        self.assertTrue(m.kkt.dummy_obj.active)
        self.assertEqual(m.kkt.dummy_obj.expr, 1.0)

    def test_solve_parametrized_kkt(self):
        m = self.get_bilevel_model()

        # test with a few values
        m.outer1.fix(1)
        m.outer2.fix(1)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(
            m_reform, parametrize_wrt=[m_reform.outer1, m_reform.outer2]
        )
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

        m.outer1.fix(1)
        m.outer2.fix(5)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(
            m_reform, parametrize_wrt=[m_reform.outer1, m_reform.outer2]
        )
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

        m.outer1.fix(3)
        m.outer2.fix(3)

        m_reform = m.clone()
        TransformationFactory('core.kkt').apply_to(
            m_reform, parametrize_wrt=[m_reform.outer1, m_reform.outer2]
        )
        TransformationFactory("mpec.simple_nonlinear").apply_to(m_reform)

        self.check_primal_kkt_transformation_solns(m, m_reform)

    def test_multiple_obj_error(self):
        m = self.get_bilevel_model()
        m.obj.deactivate()
        kkt = TransformationFactory('core.kkt')

        with self.assertRaisesRegex(
            ValueError, f"model must have only one active objective; found 0"
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
            f"""model already has an attribute with the 
                specified kkt_block_name: 'b1'""",
        ):
            kkt.apply_to(m, kkt_block_name='b1')

    def test_parametrize_wrt_unknown_error(self):
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
            "A variable passed in parametrize_wrt does not exist on an "
            "active constraint or objective within the model.",
        ):
            kkt.apply_to(m, parametrize_wrt=[m.b1.x1])

    def test_get_constraint_from_multiplier_error(self):
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
            kkt.get_constraint_from_multiplier(m, m2.gamma)

    def test_get_multiplier_from_constraint_error(self):
        m = ConcreteModel(name="model")
        m.x = Var(domain=Reals, bounds=(0, 10))
        m.y = Var(domain=Reals)
        m.obj = Objective(expr=(m.x - 3) ** 2, sense=minimize)
        m.c1 = Constraint(expr=m.x**2 + m.y**2 <= 9)
        m.c2 = Constraint(expr=(0, m.y, 10))
        kkt = TransformationFactory('core.kkt')
        kkt.apply_to(m)

        m2 = ConcreteModel()
        m2.z = Var(bounds=(1, 10))
        m2.new_con = Constraint(expr=m.x <= 5)

        with self.assertRaisesRegex(
            ValueError, "Constraint 'new_con' does not exist on model."
        ):
            kkt.get_multiplier_from_constraint(m, component=m2.new_con)

        with self.assertRaisesRegex(
            ValueError, "No multipliers exist for variable 'z' on model."
        ):
            kkt.get_multiplier_from_constraint(m, component=m2.z)
