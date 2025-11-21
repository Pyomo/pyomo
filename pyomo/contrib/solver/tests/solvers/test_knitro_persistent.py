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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.contrib.solver.solvers.knitro.persistent import KnitroPersistentSolver

avail = KnitroPersistentSolver().available()


@unittest.skipIf(not avail, "KNITRO solver is not available")
class TestKnitroPersistentSolver(unittest.TestCase):
    def setUp(self):
        self.opt = KnitroPersistentSolver()

    def test_basics(self):
        self.assertTrue(self.opt.is_persistent())
        self.assertEqual(self.opt.name, "knitro_persistent")
        self.assertTrue(self.opt.available())

    def test_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        self.opt.set_instance(m)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -1004)
        self.assertAlmostEqual(m.x.value, 5)
        self.assertAlmostEqual(m.y.value, -5)

    def test_incremental_add_variables(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        self.opt.add_variables([m.x])

        # Add variable y incrementally
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        self.opt.add_variables([m.y])

        # Add objective
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        self.opt.set_objective(m.obj)
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -1004)
        self.assertAlmostEqual(m.x.value, 5)
        self.assertAlmostEqual(m.y.value, -5)

    def test_incremental_add_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=(m.y - m.x) ** 2, sense=pyo.minimize)

        self.opt.set_instance(m)

        # Add constraint incrementally
        m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 4)
        self.opt.add_constraints([m.c1])

        results = self.opt.solve(m)
        self.assertAlmostEqual(results.incumbent_objective, 0.0)
        # Check feasibility
        self.assertTrue(pyo.value(m.x) ** 2 + pyo.value(m.y) ** 2 <= 4.0001)

    def test_incremental_add_block(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=0, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)
        self.opt.set_instance(m)

        m.b = pyo.Block()
        m.b.y = pyo.Var(initialize=0, bounds=(-5, 5))
        m.b.c = pyo.Constraint(expr=m.b.y >= m.x)

        self.opt.add_block(m.b)

        # Update objective to include y
        m.obj.expr += m.b.y
        self.opt.set_objective(m.obj)

        self.opt.solve(m)
        # min x + y s.t. y >= x, -5<=x<=5, -5<=y<=5
        # x=-5, y=-5 => obj = -10
        self.assertAlmostEqual(m.x.value, -5)
        self.assertAlmostEqual(m.b.y.value, -5)

    def test_incremental_set_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=(m.x - m.y) ** 2, sense=pyo.minimize)

        self.opt.set_objective(m.obj)

        # Add constraint incrementally
        m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 4)
        self.opt.add_constraints([m.c1])

        results = self.opt.solve(m)
        self.assertAlmostEqual(results.incumbent_objective, 0)
        # Check feasibility
        self.assertTrue(pyo.value(m.x) ** 2 + pyo.value(m.y) ** 2 <= 4.0001)