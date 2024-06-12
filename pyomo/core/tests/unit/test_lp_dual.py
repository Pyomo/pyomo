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
    ConcreteModel, 
    Constraint,
    maximize,
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
from pyomo.core.expr.compare import assertExpressionsEqual
from pyomo.opt import SolverFactory, WriterFactory


@unittest.skipUnless(scipy_available, "Scipy not available")
class TestLPDual(unittest.TestCase):
    @unittest.skipUnless(SolverFactory('gurobi').available(exception_flag=False) and
                         SolverFactory('gurobi').license_is_valid(),
                         "Gurobi is not available")
    def test_lp_dual_solve(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonPositiveReals)
        m.z = Var(domain=Reals)

        m.obj = Objective(expr=m.x + 2*m.y - 3*m.z)
        m.c1 = Constraint(expr=-4*m.x - 2*m.y - m.z <= -5)
        m.c2 = Constraint(expr=m.x + m.y <= 3)
        m.c3 = Constraint(expr=- m.y - m.z <= -4.2)
        m.c4 = Constraint(expr=m.z <= 42)
        m.dual = Suffix(direction=Suffix.IMPORT)

        lp_dual = TransformationFactory('core.lp_dual')
        dual = lp_dual.create_using(m)
        dual.dual = Suffix(direction=Suffix.IMPORT)

        opt = SolverFactory('gurobi')
        results = opt.solve(m)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)
        results = opt.solve(dual)
        self.assertEqual(results.solver.termination_condition,
                         TerminationCondition.optimal)

        self.assertAlmostEqual(value(m.obj), value(dual.obj))
        for idx, cons in enumerate([m.c1, m.c2, m.c3, m.c4]):
            self.assertAlmostEqual(value(dual.x[idx]), value(m.dual[cons]))
        # for idx, (mult, v) in enumerate([(1, m.x), (-1, m.y), (1, m.z)]):
        #     self.assertAlmostEqual(mult*value(v), value(dual.dual[dual_cons]))


    def test_lp_dual(self):
        m = ConcreteModel()
        m.x = Var(domain=NonNegativeReals)
        m.y = Var(domain=NonPositiveReals)
        m.z = Var(domain=Reals)

        m.obj = Objective(expr=m.x + 2*m.y - 3*m.z)
        m.c1 = Constraint(expr=-4*m.x - 2*m.y - m.z <= -5)
        m.c2 = Constraint(expr=m.x + m.y >= 3)
        m.c3 = Constraint(expr=- m.y - m.z == -4.2)
        m.c4 = Constraint(expr=m.z <= 42)
        m.dual = Suffix(direction=Suffix.IMPORT)

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

        assertExpressionsEqual(
            self,
            dx.expr,
            -4.0 * alpha + beta <= 1.0
        )
        assertExpressionsEqual(
            self, dy.expr, -2.0 * alpha + beta - lamb >= 2.0
        )
        assertExpressionsEqual(
            self, dz.expr, - alpha - 1.0 * lamb + xi == -3.0
        )

        dual_obj = dual.obj
        self.assertIsInstance(dual_obj, Objective)
        self.assertEqual(dual_obj.sense, maximize)
        assertExpressionsEqual(
            self, dual_obj.expr, -5 * alpha + 3 * beta - 4.2 * lamb + 42 * xi
        )
