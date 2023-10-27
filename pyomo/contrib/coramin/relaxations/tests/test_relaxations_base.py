import unittest
import pyomo.environ as pe
from pyomo.opt import assert_optimal_termination
import coramin
from pyomo.core.base.var import SimpleVar


"""
Things to test
- relaxations are valid under all possible conditions
  - continuous relaxations
  - nonlinear relaxations
  - pw relaxations
- infinite variable bounds
- tightness of relaxations
  - as variable bounds improve, the relaxations should get tighter
  - at variable bounds, relaxations should be exact
  - cuts improve relaxation
- large coef
- small coef
- safety tol
- relaxation side
- modifications to relaxations
- rebuild
  - if things don't need removed, don't remove them
  - rebuild with original constraint
- cloning
"""


class TestBaseRelaxation(unittest.TestCase):
    def test_push_and_pop_oa_points(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x, aux_var=m.y)
        m.obj = pe.Objective(expr=m.y)

        opt = pe.SolverFactory('appsi_gurobi')

        res = opt.solve(m)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, -0.5)
        self.assertAlmostEqual(m.y.value, -2)

        m.x.value = -1
        m.rel.add_cut(keep_cut=True)
        res = opt.solve(m)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, -1)

        m.rel.push_oa_points()
        m.rel.rebuild()
        res = opt.solve(m)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, -1)

        m.rel.clear_oa_points()
        m.rel.rebuild()
        res = opt.solve(m)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, -0.5)
        self.assertAlmostEqual(m.y.value, -2)

        m.x.value = -0.5
        m.rel.add_cut(keep_cut=True)
        res = opt.solve(m)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0.25)
        self.assertAlmostEqual(m.y.value, -0.5)

        m.rel.pop_oa_points()
        m.rel.rebuild()
        res = opt.solve(m)
        assert_optimal_termination(res)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, -1)

    def test_push_oa_points_with_key(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.c = coramin.relaxations.PWXSquaredRelaxation()
        m.c.build(x=m.x, aux_var=m.y)
        m.c.add_oa_point(pe.ComponentMap([(m.x, 0)]))
        self.assertEqual(list(m.c._oa_points.keys()), [(-1,), (1,), (0,)])
        m.c.push_oa_points(key='first key')
        m.c.add_oa_point(pe.ComponentMap([(m.x, 0.5)]))
        self.assertEqual(list(m.c._oa_points.keys()), [(-1,), (1,), (0,), (0.5,)])
        m.c.push_oa_points()
        m.c.add_oa_point(pe.ComponentMap([(m.x, -0.5)]))
        self.assertEqual(list(m.c._oa_points.keys()), [(-1,), (1,), (0,), (0.5,), (-0.5,)])
        m.c.push_oa_points(key='second key')
        m.c.pop_oa_points(key='first key')
        self.assertEqual(list(m.c._oa_points.keys()), [(-1,), (1,), (0,)])
        m.c.pop_oa_points()
        self.assertEqual(list(m.c._oa_points.keys()), [(-1,), (1,), (0,), (0.5,)])
        m.c.pop_oa_points(key='second key')
        self.assertEqual(list(m.c._oa_points.keys()), [(-1,), (1,), (0,), (0.5,), (-0.5,)])

    def test_push_and_pop_partitions(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var()
        m.rel = coramin.relaxations.PWXSquaredRelaxation()
        m.rel.build(x=m.x, aux_var=m.y)
        m.obj = pe.Objective(expr=m.y)
        self.assertEqual(m.rel._partitions[m.x], [-2, 1])

        m.rel.add_partition_point(-1)
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -1, 1])

        m.rel.push_partitions()
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -1, 1])

        m.rel.clear_partitions()
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, 1])

        m.rel.add_partition_point(-0.5)
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -0.5, 1])

        m.rel.pop_partitions()
        m.rel.rebuild()
        self.assertEqual(m.rel._partitions[m.x], [-2, -1, 1])

    def test_push_and_pop_partitions_2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.c = coramin.relaxations.PWXSquaredRelaxation()
        m.c.build(x=m.x, aux_var=m.y)
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [-1, 1])]))
        m.x.setlb(0)
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [0, 1])]))
        m.x.setlb(-1)
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [-1, 1])]))
        m.x.value = 0.5
        m.c.add_partition_point()
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [-1, 0.5, 1])]))
        m.x.setlb(0)
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [0, 0.5, 1])]))
        m.x.setlb(-1)
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [-1, 0.5, 1])]))
        m.x.setub(0)
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [-1, 0])]))
        m.x.setub(1)
        m.c.rebuild()
        self.assertEqual(m.c._partitions, pe.ComponentMap([(m.x, [-1, 1])]))
