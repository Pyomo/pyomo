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

from pyomo.contrib.solver.solvers.highs import Highs

opt = Highs()
if not opt.available():
    raise unittest.SkipTest


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

        opt = Highs()
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

        opt = Highs()
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

        opt = Highs()

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

    def test_qp1(self):
        # test issue #3381
        m = pyo.ConcreteModel()

        m.x1 = pyo.Var(name='x1', domain=pyo.Reals)
        m.x2 = pyo.Var(name='x2', domain=pyo.Reals)

        # Quadratic Objective function
        m.obj = pyo.Objective(expr=m.x1 * m.x1 + m.x2 * m.x2, sense=pyo.minimize)

        m.con1 = pyo.Constraint(expr=m.x1 >= 1)
        m.con2 = pyo.Constraint(expr=m.x2 >= 1)

        results = opt.solve(m)
        self.assertAlmostEqual(m.x1.value, 1, places=5)
        self.assertAlmostEqual(m.x2.value, 1, places=5)
        self.assertEqual(results.objective_bound, 2)

    def test_qp2(self):
        # test issue #3381
        m = pyo.ConcreteModel()

        m.x1 = pyo.Var(name='x1', domain=pyo.Reals)
        m.x2 = pyo.Var(name='x2', domain=pyo.Reals)

        m.p = pyo.Param(initialize=1, mutable=True)

        m.obj = pyo.Objective(
            expr=m.p * m.x1 * m.x1 + m.x2 * m.x2 - m.x1 * m.x2, sense=pyo.minimize
        )

        m.con1 = pyo.Constraint(expr=m.x1 >= 1)
        m.con2 = pyo.Constraint(expr=m.x2 >= 1)

        opt.set_instance(m)
        results = opt.solve(m)
        self.assertAlmostEqual(m.x1.value, 1, places=5)
        self.assertAlmostEqual(m.x2.value, 1, places=5)
        self.assertEqual(results.objective_bound, 1)

        m.p.value = 2.0
        opt.update_parameters()
        results = opt.solve(m)
        self.assertAlmostEqual(m.x1.value, 1, places=5)
        self.assertAlmostEqual(m.x2.value, 1, places=5)
        self.assertEqual(results.objective_bound, 2)
