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

from pyomo.common.dependencies import scipy_available
import pyomo.common.unittest as unittest
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    maximize,
    minimize,
    NonNegativeReals,
    NonPositiveReals,
    Objective,
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
        m.dual = Suffix(direction=Suffix.IMPORT)

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m)
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

    def test_parameterized_linear_dual(self):
        m = ConcreteModel()

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
