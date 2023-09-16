import pyomo.common.unittest as unittest
import pyomo.environ as pe
from pyomo.contrib.appsi.solvers.highs import Highs
from pyomo.contrib.appsi.base import TerminationCondition


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
        self.assertAlmostEqual(res.best_feasible_objective, 1)

        del m.c1
        m.p2.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -8)

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
        self.assertAlmostEqual(res.best_feasible_objective, 1)

        del m.c1
        del m.c2
        m.p1.value = -9
        m.p2.value = 9
        res = opt.solve(m)
        self.assertAlmostEqual(res.best_feasible_objective, -9)
