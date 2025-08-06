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

from pyomo.common.dependencies import scipy_available
import pyomo.common.unittest as unittest
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    maximize,
    minimize,
    NonNegativeIntegers,
    NonNegativeReals,
    NonPositiveReals,
    Objective,
    Param,
    Reals,
    Suffix,
    TerminationCondition,
    TransformationFactory,
    value,
    Var,
)
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
from pyomo.opt import SolverFactory, WriterFactory


@unittest.skipUnless(scipy_available, "Scipy not available")
class TestLPDual(unittest.TestCase):
    def check_primal_dual_solns(self, m, dual):
        lp_dual = TransformationFactory('core.lp_dual')

        m.dual = Suffix(direction=Suffix.IMPORT)
        dual.dual = Suffix(direction=Suffix.IMPORT)

        opt = SolverFactory('gurobi')
        results = opt.solve(m)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        results = opt.solve(dual)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )

        self.assertAlmostEqual(value(m.obj), value(dual.obj))
        for cons in [m.c1, m.c2, m.c3, m.c4]:
            self.assertAlmostEqual(
                value(lp_dual.get_dual_var(dual, cons)), value(m.dual[cons])
            )
        for v in [m.x, m.y, m.z]:
            self.assertAlmostEqual(
                value(v), value(dual.dual[lp_dual.get_dual_constraint(dual, v)])
            )

    @unittest.skipUnless(
        SolverFactory('gurobi').available(exception_flag=False)
        and SolverFactory('gurobi').license_is_valid(),
        "Gurobi is not available",
    )
    def test_lp_dual_solve(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonPositiveReals)
        m.z = Var(domain=Reals)

        m.obj = Objective(expr=m.x + 2 * m.y - 3 * m.z)
        m.c1 = Constraint(expr=-4 * m.x - 2 * m.y - m.z <= -5)
        m.c2 = Constraint(expr=m.x + m.y <= 3)
        m.c3 = Constraint(expr=-m.y - m.z <= -4.2)
        m.c4 = Constraint(expr=m.z <= 42)

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m)

        self.check_primal_dual_solns(m, dual)

    def test_lp_dual(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonPositiveReals)
        m.z = Var(domain=Reals)

        m.obj = Objective(expr=m.x + 2 * m.y - 3 * m.z)
        m.c1 = Constraint(expr=-4 * m.x - 2 * m.y - m.z <= -5)
        m.c2 = Constraint(expr=m.x + m.y >= 3)
        m.c3 = Constraint(expr=-m.y - m.z == -4.2)
        m.c4 = Constraint(expr=m.z <= 42)

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m)

        alpha = lp_dual.get_dual_var(dual, m.c1)
        beta = lp_dual.get_dual_var(dual, m.c2)
        lamb = lp_dual.get_dual_var(dual, m.c3)
        xi = lp_dual.get_dual_var(dual, m.c4)

        self.assertIs(lp_dual.get_primal_constraint(dual, alpha), m.c1)
        self.assertIs(lp_dual.get_primal_constraint(dual, beta), m.c2)
        self.assertIs(lp_dual.get_primal_constraint(dual, lamb), m.c3)
        self.assertIs(lp_dual.get_primal_constraint(dual, xi), m.c4)

        dx = lp_dual.get_dual_constraint(dual, m.x)
        dy = lp_dual.get_dual_constraint(dual, m.y)
        dz = lp_dual.get_dual_constraint(dual, m.z)

        self.assertIs(lp_dual.get_primal_var(dual, dx), m.x)
        self.assertIs(lp_dual.get_primal_var(dual, dy), m.y)
        self.assertIs(lp_dual.get_primal_var(dual, dz), m.z)

        self.assertIs(alpha.ctype, Var)
        self.assertEqual(alpha.domain, NonPositiveReals)
        self.assertEqual(alpha.ub, 0)
        self.assertIsNone(alpha.lb)
        self.assertIs(beta.ctype, Var)
        self.assertEqual(beta.domain, NonNegativeReals)
        self.assertEqual(beta.lb, 0)
        self.assertIsNone(beta.ub)
        self.assertIs(lamb.ctype, Var)
        self.assertEqual(lamb.domain, Reals)
        self.assertIsNone(lamb.ub)
        self.assertIsNone(lamb.lb)
        self.assertIs(xi.ctype, Var)
        self.assertEqual(xi.domain, NonPositiveReals)
        self.assertEqual(xi.ub, 0)
        self.assertIsNone(xi.lb)

        self.assertIs(dx.ctype, Constraint)
        self.assertIs(dy.ctype, Constraint)
        self.assertIs(dz.ctype, Constraint)

        assertExpressionsStructurallyEqual(self, dx.expr, -4.0 * alpha + beta <= 1.0)
        assertExpressionsStructurallyEqual(
            self, dy.expr, -2.0 * alpha + beta - lamb >= 2.0
        )
        assertExpressionsStructurallyEqual(
            self, dz.expr, -alpha - 1.0 * lamb + xi == -3.0
        )

        dual_obj = dual.obj
        self.assertIsInstance(dual_obj, Objective)
        self.assertEqual(dual_obj.sense, maximize)
        assertExpressionsEqual(
            self, dual_obj.expr, -5 * alpha + 3 * beta - 4.2 * lamb + 42 * xi
        )

        ##
        # now go the other way and recover the primal
        ##

        primal = lp_dual.create_using(dual)

        x = lp_dual.get_dual_var(primal, dx)
        y = lp_dual.get_dual_var(primal, dy)
        z = lp_dual.get_dual_var(primal, dz)

        self.assertIs(x.ctype, Var)
        self.assertEqual(x.domain, NonNegativeReals)
        self.assertEqual(x.lb, 0)
        self.assertIsNone(x.ub)
        self.assertIs(y.ctype, Var)
        self.assertEqual(y.domain, NonPositiveReals)
        self.assertIsNone(y.lb)
        self.assertEqual(y.ub, 0)
        self.assertIs(z.ctype, Var)
        self.assertEqual(z.domain, Reals)
        self.assertIsNone(z.lb)
        self.assertIsNone(z.ub)

        self.assertIs(lp_dual.get_primal_constraint(primal, x), dx)
        self.assertIs(lp_dual.get_primal_constraint(primal, y), dy)
        self.assertIs(lp_dual.get_primal_constraint(primal, z), dz)

        dalpha = lp_dual.get_dual_constraint(primal, alpha)
        dbeta = lp_dual.get_dual_constraint(primal, beta)
        dlambda = lp_dual.get_dual_constraint(primal, lamb)
        dxi = lp_dual.get_dual_constraint(primal, xi)

        self.assertIs(lp_dual.get_primal_var(primal, dalpha), alpha)
        self.assertIs(lp_dual.get_primal_var(primal, dbeta), beta)
        self.assertIs(lp_dual.get_primal_var(primal, dlambda), lamb)
        self.assertIs(lp_dual.get_primal_var(primal, dxi), xi)

        self.assertIs(dalpha.ctype, Constraint)
        self.assertIs(dbeta.ctype, Constraint)
        self.assertIs(dlambda.ctype, Constraint)
        self.assertIs(dxi.ctype, Constraint)

        assertExpressionsStructurallyEqual(
            self, dalpha.expr, -4.0 * x - 2.0 * y - z <= -5.0
        )
        assertExpressionsStructurallyEqual(self, dbeta.expr, x + y >= 3.0)
        assertExpressionsStructurallyEqual(self, dlambda.expr, -y - z == -4.2)
        assertExpressionsStructurallyEqual(self, dxi.expr, z <= 42.0)

        primal_obj = primal.obj
        self.assertIsInstance(primal_obj, Objective)
        self.assertEqual(primal_obj.sense, minimize)
        assertExpressionsEqual(self, primal_obj.expr, x + 2.0 * y - 3.0 * z)

    def get_bilevel_model(self):
        m = ConcreteModel(name='primal')

        m.outer1 = Var(domain=Binary)
        m.outer = Var([2, 3], domain=Binary)

        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonPositiveReals)
        m.z = Var(domain=Reals)

        m.obj = Objective(expr=m.x + 2 * m.y - 3 * m.outer[3] * m.z)
        m.c1 = Constraint(expr=-4 * m.x - 2 * m.y - m.z <= -5 * m.outer1)
        m.c2 = Constraint(expr=m.x + m.outer[2] * m.y >= 3)
        m.c3 = Constraint(expr=-m.y - m.z == -4.2)
        m.c4 = Constraint(expr=m.z <= 42)

        return m

    def test_parameterized_linear_dual(self):
        m = self.get_bilevel_model()

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        alpha = lp_dual.get_dual_var(dual, m.c1)
        beta = lp_dual.get_dual_var(dual, m.c2)
        lamb = lp_dual.get_dual_var(dual, m.c3)
        mu = lp_dual.get_dual_var(dual, m.c4)

        self.assertIs(lp_dual.get_primal_constraint(dual, alpha), m.c1)
        self.assertIs(lp_dual.get_primal_constraint(dual, beta), m.c2)
        self.assertIs(lp_dual.get_primal_constraint(dual, lamb), m.c3)
        self.assertIs(lp_dual.get_primal_constraint(dual, mu), m.c4)

        dx = lp_dual.get_dual_constraint(dual, m.x)
        dy = lp_dual.get_dual_constraint(dual, m.y)
        dz = lp_dual.get_dual_constraint(dual, m.z)

        self.assertIs(lp_dual.get_primal_var(dual, dx), m.x)
        self.assertIs(lp_dual.get_primal_var(dual, dy), m.y)
        self.assertIs(lp_dual.get_primal_var(dual, dz), m.z)

        self.assertIs(alpha.ctype, Var)
        self.assertEqual(alpha.domain, NonPositiveReals)
        self.assertEqual(alpha.ub, 0)
        self.assertIsNone(alpha.lb)
        self.assertIs(beta.ctype, Var)
        self.assertEqual(beta.domain, NonNegativeReals)
        self.assertEqual(beta.lb, 0)
        self.assertIsNone(beta.ub)
        self.assertIs(lamb.ctype, Var)
        self.assertEqual(lamb.domain, Reals)
        self.assertIsNone(lamb.ub)
        self.assertIsNone(lamb.lb)
        self.assertIs(mu.ctype, Var)
        self.assertEqual(mu.domain, NonPositiveReals)
        self.assertEqual(mu.ub, 0)
        self.assertIsNone(mu.lb)

        self.assertIs(dx.ctype, Constraint)
        self.assertIs(dy.ctype, Constraint)
        self.assertIs(dz.ctype, Constraint)

        assertExpressionsStructurallyEqual(self, dx.expr, -4.0 * alpha + beta <= 1.0)
        assertExpressionsStructurallyEqual(
            self, dy.expr, -2.0 * alpha + m.outer[2] * beta - lamb >= 2.0
        )
        assertExpressionsStructurallyEqual(
            self, dz.expr, -alpha - 1.0 * lamb + mu == -3.0 * m.outer[3]
        )

        dual_obj = dual.obj
        self.assertIsInstance(dual_obj, Objective)
        self.assertEqual(dual_obj.sense, maximize)
        assertExpressionsEqual(
            self, dual_obj.expr, -5 * m.outer1 * alpha + 3 * beta - 4.2 * lamb + 42 * mu
        )

    @unittest.skipUnless(
        SolverFactory('gurobi').available(exception_flag=False)
        and SolverFactory('gurobi').license_is_valid(),
        "Gurobi is not available",
    )
    def test_solve_parameterized_lp_dual(self):
        m = self.get_bilevel_model()

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        # We just check half of the possible permutations since we're calling a
        # solver twice for all of these:
        m.outer1.fix(1)
        m.outer[2].fix(1)
        m.outer[3].fix(1)

        self.check_primal_dual_solns(m, dual)

        m.outer1.fix(0)
        m.outer[2].fix(1)
        m.outer[3].fix(0)

        self.check_primal_dual_solns(m, dual)

        m.outer1.fix(0)
        m.outer[2].fix(0)
        m.outer[3].fix(0)

        self.check_primal_dual_solns(m, dual)

        m.outer1.fix(1)
        m.outer[2].fix(1)
        m.outer[3].fix(0)

        self.check_primal_dual_solns(m, dual)

    def test_multiple_obj_error(self):
        m = self.get_bilevel_model()
        m.obj.deactivate()

        lp_dual = TransformationFactory('core.lp_dual')

        with self.assertRaisesRegex(
            ValueError,
            "Model 'primal' has no objective or multiple active objectives. "
            "Can only take dual with exactly one active objective!",
        ):
            dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        m.obj.activate()
        m.obj2 = Objective(expr=m.outer1 + m.outer[3])

        with self.assertRaisesRegex(
            ValueError,
            "Model 'primal' has no objective or multiple active objectives. "
            "Can only take dual with exactly one active objective!",
        ):
            dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

    def test_primal_constraint_map_error(self):
        m = self.get_bilevel_model()

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        with self.assertRaisesRegex(
            ValueError,
            "It does not appear that Var 'x' is a dual variable on model "
            "'primal dual'",
        ):
            thing = lp_dual.get_primal_constraint(dual, m.x)

    def test_dual_constraint_map_error(self):
        m = self.get_bilevel_model()

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        with self.assertRaisesRegex(
            ValueError,
            "It does not appear that Var 'outer1' is a primal variable on model "
            "'primal'",
        ):
            thing = lp_dual.get_dual_constraint(m, m.outer1)

    def test_primal_var_map_error(self):
        m = self.get_bilevel_model()

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        with self.assertRaisesRegex(
            ValueError,
            "It does not appear that Constraint 'c1' is a dual constraint "
            "on model 'primal dual'",
        ):
            thing = lp_dual.get_primal_var(dual, m.c1)

    def test_dual_var_map_error(self):
        m = self.get_bilevel_model()

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m, parameterize_wrt=[m.outer1, m.outer])

        m.c_new = Constraint(expr=m.x + m.y <= 35)

        with self.assertRaisesRegex(
            ValueError,
            "It does not appear that Constraint 'c_new' is a primal constraint "
            "on model 'primal'",
        ):
            thing = lp_dual.get_dual_var(m, m.c_new)

    def test_parameterization_makes_constraint_trivial(self):
        m = self.get_bilevel_model()
        m.budgetish = Constraint(expr=m.outer[2] + m.outer[3] == 1)

        lp_dual = TransformationFactory('core.lp_dual')
        with self.assertRaisesRegex(
            ValueError,
            "The primal model contains a constraint that the "
            "parameterization makes trivial: 'budgetish'"
            "\nPlease deactivate it or declare it on another Block "
            "before taking the dual.",
        ):
            dual = lp_dual.create_using(m, parameterize_wrt=m.outer)

    def test_normal_trivial_constraint_error(self):
        m = ConcreteModel()
        m.p = Param(initialize=3, mutable=True)
        m.x = Var(bounds=(0, 9))
        m.c = Constraint(expr=m.x * m.p <= 8)
        m.x.fix(2)

        m.obj = Objective(expr=m.x)

        lp_dual = TransformationFactory('core.lp_dual')
        with self.assertRaisesRegex(
            ValueError,
            "Model 'unknown' has no variables in the active Constraints "
            "or Objective.",
        ):
            dual = lp_dual.create_using(m)

    def test_discrete_primal_var_error(self):
        m = self.get_bilevel_model()
        m.x.domain = NonNegativeIntegers

        with self.assertRaisesRegex(
            ValueError, "The domain of the primal variable 'x' is not continuous"
        ):
            dual = TransformationFactory('core.lp_dual').create_using(
                m, parameterize_wrt=[m.outer, m.outer1]
            )
