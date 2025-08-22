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
import pyomo.environ as pyo
import pyomo.contrib.solver.solvers.knitro as knitro

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
            "NAME",
            "solve",
            "version",
        ]
        method_list = [m for m in dir(opt) if not m.startswith("_")]
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
    def create_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        model.y = pyo.Var(initialize=1.5, bounds=(-5, 5))

        def dummy_equation(m):
            return (1.0 - m.x) + 100.0 * (m.y - m.x)

        model.obj = pyo.Objective(rule=dummy_equation, sense=pyo.minimize)
        return model

    def create_qp_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        model.y = pyo.Var(initialize=1.5, bounds=(-5, 5))

        def dummy_qp_equation(m):
            return (1.0 - m.x) + 100.0 * (m.y - m.x) ** 2

        model.obj = pyo.Objective(rule=dummy_qp_equation, sense=pyo.minimize)
        return model

    def create_qcp_model(self):
        model = pyo.ConcreteModel()
        model.x = pyo.Var(initialize=1.5, bounds=(-5, 5))
        model.y = pyo.Var(initialize=1.5, bounds=(-5, 5))

        def dummy_qcp_equation(m):
            return (m.y - m.x) ** 2

        def dummy_qcp_constraint(m):
            return m.x**2 + m.y**2 <= 4

        model.obj = pyo.Objective(rule=dummy_qcp_equation, sense=pyo.minimize)
        model.c1 = pyo.Constraint(rule=dummy_qcp_constraint)
        return model

    def test_solve(self):
        model = self.create_model()
        opt = knitro.KnitroDirectSolver()
        opt.solve(model)
        self.assertAlmostEqual(model.x.value, 5)
        self.assertAlmostEqual(model.y.value, -5)

    def test_qp_solve(self):
        model = self.create_qp_model()
        opt = knitro.KnitroDirectSolver()
        results = opt.solve(model)
        self.assertAlmostEqual(results.incumbent_objective, -4.0, 3)
        self.assertAlmostEqual(model.x.value, 5.0, 3)
        self.assertAlmostEqual(model.y.value, 5.0, 3)

    def test_qcp_solve(self):
        model = self.create_qcp_model()
        opt = knitro.KnitroDirectSolver()
        results = opt.solve(model)
        self.assertAlmostEqual(results.incumbent_objective, 0.0)
