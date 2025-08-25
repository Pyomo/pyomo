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

import unittest

import pyomo.contrib.solver.solvers.knitro as knitro
import pyomo.environ as pyo

avail = knitro.KnitroDirectSolver().available()


@unittest.skipIf(not avail, "KNITRO solver is not available")
class TestKnitroDirectSolverConfig(unittest.TestCase):
    def test_default_instantiation(self):
        config = knitro.KnitroConfig()
        self.assertIsNone(config._description)
        self.assertEqual(config._visibility, 0)
        self.assertFalse(config.tee)
        self.assertTrue(config.load_solutions)
        self.assertTrue(config.raise_exception_on_nonoptimal_result)
        self.assertFalse(config.symbolic_solver_labels)
        self.assertIsNone(config.timer)
        self.assertIsNone(config.threads)
        self.assertIsNone(config.time_limit)

    def test_custom_instantiation(self):
        config = knitro.KnitroConfig(description="A description")
        config.tee = True
        self.assertTrue(config.tee)
        self.assertEqual(config._description, "A description")
        self.assertIsNone(config.time_limit)


@unittest.skipIf(not avail, "KNITRO solver is not available")
class TestKnitroDirectSolverInterface(unittest.TestCase):
    def test_class_member_list(self):
        opt = knitro.KnitroDirectSolver()
        expected_list = [
            "CONFIG",
            "available",
            "config",
            "api_version",
            "is_persistent",
            "name",
            "solve",
            "version",
        ]
        method_list = [
            m for m in dir(opt) if not m.startswith("_") and not m.startswith("get")
        ]
        self.assertListEqual(sorted(method_list), sorted(expected_list))

    def test_default_instantiation(self):
        opt = knitro.KnitroDirectSolver()
        self.assertFalse(opt.is_persistent())
        self.assertIsNotNone(opt.version())
        self.assertEqual(opt.name, "knitro_direct")
        self.assertEqual(opt.CONFIG, opt.config)
        self.assertTrue(opt.available())

    def test_instantiation_as_context(self):
        with knitro.KnitroDirectSolver() as opt:
            self.assertFalse(opt.is_persistent())
            self.assertIsNotNone(opt.version())
            self.assertEqual(opt.name, "knitro_direct")
            self.assertEqual(opt.CONFIG, opt.config)
            self.assertTrue(opt.available())

    def test_available_cache(self):
        opt = knitro.KnitroDirectSolver()
        opt.available()
        self.assertTrue(opt._available_cache)
        self.assertIsNotNone(opt._available_cache)


class TestKnitroDirectSolver(unittest.TestCase):
    def setUp(self):
        self.opt = knitro.KnitroDirectSolver()

    def test_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x), sense=pyo.minimize
        )
        res = self.opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -1004)
        self.assertAlmostEqual(m.x.value, 5)
        self.assertAlmostEqual(m.y.value, -5)

    def test_qp_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(
            expr=(1.0 - m.x) + 100.0 * (m.y - m.x) ** 2, sense=pyo.minimize
        )
        results = self.opt.solve(m)
        self.assertAlmostEqual(results.incumbent_objective, -4.0, 3)
        self.assertAlmostEqual(m.x.value, 5.0, 3)
        self.assertAlmostEqual(m.y.value, 5.0, 3)

    def test_qcp_solve(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.y = pyo.Var(initialize=1.5, bounds=(-5, 5))
        m.obj = pyo.Objective(expr=(m.y - m.x) ** 2, sense=pyo.minimize)
        m.c1 = pyo.Constraint(expr=m.x**2 + m.y**2 <= 4)
        results = self.opt.solve(m)
        self.assertAlmostEqual(results.incumbent_objective, 0.0)

    def test_solve_exp(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= pyo.exp(m.x))
        self.opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.42630274815985264)
        self.assertAlmostEqual(m.y.value, 0.6529186341994245)

    def test_solve_log(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1)
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y <= pyo.log(m.x))
        self.opt.solve(m)
        self.assertAlmostEqual(m.x.value, 0.6529186341994245)
        self.assertAlmostEqual(m.y.value, -0.42630274815985264)
