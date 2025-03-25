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

import subprocess
import sys

import pyomo.common.unittest as unittest
import pyomo.environ as pe

from pyomo.contrib.solver.solvers.highs import Highs
from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output

opt = Highs()
if not opt.available():
    raise unittest.SkipTest


class TestBugs(unittest.TestCase):
    def test_mutable_params_with_remove_cons(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-10, 10))
        m.y = pe.Var()

        m.p1 = pe.Param(mutable=True)
        m.p2 = pe.Param(mutable=True)

        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + m.p1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + m.p2)

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
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()

        m.p1 = pe.Param(mutable=True)
        m.p2 = pe.Param(mutable=True)

        m.y.setlb(m.p1)
        m.y.setub(m.p2)

        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)

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

        m = pe.ConcreteModel()
        m.x = pe.Var(domain=pe.Binary)
        m.y = pe.Var(domain=pe.Binary)
        m.fx = pe.Var(domain=pe.NonNegativeReals)
        m.fy = pe.Var(domain=pe.NonNegativeReals)
        m.c1 = pe.Constraint(expr=m.fx <= m.x)
        m.c2 = pe.Constraint(expr=m.fy <= m.y)
        m.c3 = pe.Constraint(expr=m.x + m.y <= 1)

        m.obj = pe.Objective(expr=m.fx * 0.5 + m.fy * 0.4, sense=pe.maximize)

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

    def test_qp(self):
        # test issue #3381
        m = pe.ConcreteModel()

        m.x1 = pe.Var(name='x1', domain=pe.Reals)
        m.x2 = pe.Var(name='x2', domain=pe.Reals)

        # Quadratic Objective function
        m.obj = pe.Objective(expr=m.x1 * m.x1 + m.x2 * m.x2, sense=pe.minimize)

        m.con1 = pe.Constraint(expr=m.x1 >= 1)
        m.con2 = pe.Constraint(expr=m.x2 >= 1)

        results = opt.solve(m)
        self.assertAlmostEqual(m.x1.value, 1, places=5)
        self.assertAlmostEqual(m.x2.value, 1, places=5)
        self.assertEqual(results.objective_bound, 2)
