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

import pyomo.contrib.solver.solvers.highs as highs
from pyomo.contrib.solver.common.base import Availability
from pyomo.contrib.solver.common.results import SolutionStatus

highs_available = highs.Highs().available()


@unittest.skipIf(not highs_available, "highspy is not available")
class TestHighsInterface(unittest.TestCase):
    def test_default_instantiation(self):
        opt = highs.Highs()
        self.assertTrue(opt.is_persistent())
        self.assertEqual(opt.name, 'highs')
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertTrue(opt.available())
        self.assertIsNotNone(opt.version())

    def test_available_cache_and_recheck(self):
        opt = highs.Highs()

        first = opt.available()
        self.assertTrue(first)
        self.assertIsNotNone(opt._available_cache)

        # Make sure that recheck works by faking highspy_available
        with unittest.mock.patch(
            'pyomo.contrib.solver.solvers.highs.highspy_available', False
        ):
            self.assertEqual(opt.available(), first)
            self.assertEqual(opt.available(recheck=True), Availability.NotFound)

    def test_version_cache_and_recheck_with_attrs(self):
        opt = highs.Highs()
        version = opt.version()
        self.assertIsNotNone(version)
        self.assertIsNotNone(opt._version_cache)


@unittest.skipIf(not highs_available, "highspy is not available")
class TestHighs(unittest.TestCase):
    def create_lp_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.con = pyo.Constraint(expr=m.x + m.y >= 1)
        m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
        return m

    def create_mip_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.con1 = pyo.Constraint(expr=m.x + m.y >= 3)
        m.con2 = pyo.Constraint(expr=m.y <= 2)
        m.obj = pyo.Objective(expr=3 * m.x + m.y, sense=pyo.minimize)
        return m

    def test_lp_solve(self):
        m = self.create_lp_model()
        res = highs.Highs().solve(m)
        self.assertEqual(res.solver_name, 'highs')
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(pyo.value(m.obj), 1.0, places=7)
        self.assertGreaterEqual(m.x.value, 0.0)
        self.assertGreaterEqual(m.y.value, 0.0)
        self.assertGreaterEqual(m.x.value + m.y.value, 1.0 - 1e-7)

    def test_mip_solve(self):
        m = self.create_mip_model()
        res = highs.Highs().solve(m)
        self.assertEqual(res.solver_name, 'highs')
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(pyo.value(m.obj), 5.0, places=7)
        self.assertAlmostEqual(m.x.value, 1.0, places=7)
        self.assertAlmostEqual(m.y.value, 2.0, places=7)

    def test_persistent_update_path(self):
        m = self.create_lp_model()
        opt = highs.Highs()
        res1 = opt.solve(m)
        self.assertEqual(res1.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(pyo.value(m.obj), 1.0, places=7)

        # Tighten the constraint
        m.con.set_value(m.x + m.y >= 1.5)
        res2 = opt.solve(m)
        self.assertEqual(res2.solution_status, SolutionStatus.optimal)
        self.assertAlmostEqual(pyo.value(m.obj), 1.5, places=7)


@unittest.skipIf(not highs_available, "highspy is not available")
class TestBugs(unittest.TestCase):
    def test_mutable_params_with_remove_cons(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var()

        m.p1 = pyo.Param(mutable=True)
        m.p2 = pyo.Param(mutable=True)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x + m.p1)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x + m.p2)

        m.p1.value = 1
        m.p2.value = 1

        opt = highs.Highs()
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, 1)

        del m.c1
        m.p2.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, -8)

    def test_mutable_params_with_remove_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()

        m.p1 = pyo.Param(mutable=True)
        m.p2 = pyo.Param(mutable=True)

        m.y.setlb(m.p1)
        m.y.setub(m.p2)

        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)

        m.p1.value = -10
        m.p2.value = 10

        opt = highs.Highs()
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, 1)

        del m.c1
        del m.c2
        m.p1.value = -9
        m.p2.value = 9
        res = opt.solve(m)
        self.assertAlmostEqual(res.objective_bound, -9)

    def test_fix_and_unfix(self):
        # Tests issue https://github.com/Pyomo/pyomo/issues/3127

        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.Binary)
        m.y = pyo.Var(domain=pyo.Binary)
        m.fx = pyo.Var(domain=pyo.NonNegativeReals)
        m.fy = pyo.Var(domain=pyo.NonNegativeReals)
        m.c1 = pyo.Constraint(expr=m.fx <= m.x)
        m.c2 = pyo.Constraint(expr=m.fy <= m.y)
        m.c3 = pyo.Constraint(expr=m.x + m.y <= 1)

        m.obj = pyo.Objective(expr=m.fx * 0.5 + m.fy * 0.4, sense=pyo.maximize)

        opt = highs.Highs()

        # solution 1 has m.x == 1 and m.y == 0
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 1, places=5)
        self.assertAlmostEqual(m.fy.value, 0, places=5)
        self.assertAlmostEqual(r.objective_bound, 0.5, places=5)

        # solution 2 has m.x == 0 and m.y == 1
        m.y.fix(1)
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 0, places=5)
        self.assertAlmostEqual(m.fy.value, 1, places=5)
        self.assertAlmostEqual(r.objective_bound, 0.4, places=5)

        # solution 3 should be equal solution 1
        m.y.unfix()
        m.x.fix(1)
        r = opt.solve(m)
        self.assertAlmostEqual(m.fx.value, 1, places=5)
        self.assertAlmostEqual(m.fy.value, 0, places=5)
        self.assertAlmostEqual(r.objective_bound, 0.5, places=5)
