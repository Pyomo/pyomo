import unittest
import math
import pyomo.environ as pe
import coramin
import numpy as np
from coramin.relaxations.segments import compute_k_segment_points


class TestUnivariateExp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = pe.ConcreteModel()
        cls.model = model
        model.y = pe.Var()
        model.x = pe.Var(bounds=(-1.5, 1.5))

        model.obj = pe.Objective(expr=model.y, sense=pe.maximize)
        model.pw_exp = coramin.relaxations.PWUnivariateRelaxation()
        model.pw_exp.build(x=model.x, aux_var=model.y, pw_repn='INC', shape=coramin.utils.FunctionShape.CONVEX,
                           relaxation_side=coramin.utils.RelaxationSide.BOTH, f_x_expr=pe.exp(model.x))
        model.pw_exp.add_partition_point(-0.5)
        model.pw_exp.add_partition_point(0.5)
        model.pw_exp.rebuild()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_exp_ub(self):
        model = self.model.clone()

        solver = pe.SolverFactory('gurobi_direct')
        solver.solve(model)
        self.assertAlmostEqual(pe.value(model.y), math.exp(1.5), 4)

    def test_exp_mid(self):
        model = self.model.clone()
        model.x_con = pe.Constraint(expr=model.x <= 0.3)

        solver = pe.SolverFactory('gurobi_direct')
        solver.solve(model)
        self.assertAlmostEqual(pe.value(model.y), 1.44, 3)

    def test_exp_lb(self):
        model = self.model.clone()
        model.obj.sense = pe.minimize

        solver = pe.SolverFactory('gurobi_direct')
        solver.solve(model)
        self.assertAlmostEqual(pe.value(model.y), math.exp(-1.5), 4)


class TestUnivariate(unittest.TestCase):
    def helper(self, func, shape, bounds_list, relaxation_class, relaxation_side=coramin.utils.RelaxationSide.BOTH):
        for lb, ub in bounds_list:
            num_segments_list = [1, 2, 3]
            m = pe.ConcreteModel()
            m.x = pe.Var(bounds=(lb, ub))
            m.aux = pe.Var()
            if relaxation_class is coramin.relaxations.PWUnivariateRelaxation:
                m.c = coramin.relaxations.PWUnivariateRelaxation()
                m.c.build(x=m.x,
                          aux_var=m.aux,
                          relaxation_side=relaxation_side,
                          shape=shape,
                          f_x_expr=func(m.x))
            else:
                m.c = relaxation_class()
                m.c.build(x=m.x, aux_var=m.aux, relaxation_side=relaxation_side)
            m.p = pe.Param(mutable=True, initialize=0)
            m.c2 = pe.Constraint(expr=m.x == m.p)
            opt = pe.SolverFactory('gurobi_persistent')
            for num_segments in num_segments_list:
                segment_points = compute_k_segment_points(m.x, num_segments)
                m.c.clear_partitions()
                for pt in segment_points:
                    m.c.add_partition_point(pt)
                    var_values = pe.ComponentMap()
                    var_values[m.x] = pt
                    m.c.add_oa_point(var_values=var_values)
                m.c.rebuild()
                opt.set_instance(m)
                for _x in [float(i) for i in np.linspace(lb, ub, 10)]:
                    m.p.value = _x
                    opt.remove_constraint(m.c2)
                    opt.add_constraint(m.c2)
                    if relaxation_side in {coramin.utils.RelaxationSide.BOTH, coramin.utils.RelaxationSide.UNDER}:
                        m.obj = pe.Objective(expr=m.aux)
                        opt.set_objective(m.obj)
                        res = opt.solve()
                        self.assertEqual(res.solver.termination_condition, pe.TerminationCondition.optimal)
                        self.assertLessEqual(m.aux.value, func(_x) + 1e-10)
                        del m.obj
                    if relaxation_side in {coramin.utils.RelaxationSide.BOTH, coramin.utils.RelaxationSide.OVER}:
                        m.obj = pe.Objective(expr=m.aux, sense=pe.maximize)
                        opt.set_objective(m.obj)
                        res = opt.solve()
                        self.assertEqual(res.solver.termination_condition, pe.TerminationCondition.optimal)
                        self.assertGreaterEqual(m.aux.value, func(_x) - 1e-10)
                        del m.obj

    def test_exp(self):
        self.helper(func=pe.exp, shape=coramin.utils.FunctionShape.CONVEX, bounds_list=[(-1, 1)],
                    relaxation_class=coramin.relaxations.PWUnivariateRelaxation)

    def test_log(self):
        self.helper(func=pe.log, shape=coramin.utils.FunctionShape.CONCAVE, bounds_list=[(0.5, 1.5)],
                    relaxation_class=coramin.relaxations.PWUnivariateRelaxation)

    def test_quadratic(self):
        def quadratic_func(x):
            return x**2
        self.helper(func=quadratic_func, shape=None, bounds_list=[(-1, 2)],
                    relaxation_class=coramin.relaxations.PWXSquaredRelaxation)

    def test_arctan(self):
        self.helper(func=pe.atan, shape=None, bounds_list=[(-1, 1), (-1, 0), (0, 1)],
                    relaxation_class=coramin.relaxations.PWArctanRelaxation)
        self.helper(func=pe.atan, shape=None, bounds_list=[(-0.1, 1)],
                    relaxation_class=coramin.relaxations.PWArctanRelaxation,
                    relaxation_side=coramin.utils.RelaxationSide.OVER)
        self.helper(func=pe.atan, shape=None, bounds_list=[(-1, 0.1)],
                    relaxation_class=coramin.relaxations.PWArctanRelaxation,
                    relaxation_side=coramin.utils.RelaxationSide.UNDER)

    def test_sin(self):
        self.helper(func=pe.sin, shape=None, bounds_list=[(-1, 1), (-1, 0), (0, 1)],
                    relaxation_class=coramin.relaxations.PWSinRelaxation)
        self.helper(func=pe.sin, shape=None, bounds_list=[(-0.1, 1)],
                    relaxation_class=coramin.relaxations.PWSinRelaxation,
                    relaxation_side=coramin.utils.RelaxationSide.OVER)
        self.helper(func=pe.sin, shape=None, bounds_list=[(-1, 0.1)],
                    relaxation_class=coramin.relaxations.PWSinRelaxation,
                    relaxation_side=coramin.utils.RelaxationSide.UNDER)

    def test_cos(self):
        self.helper(func=pe.cos, shape=None, bounds_list=[(-1, 1)],
                    relaxation_class=coramin.relaxations.PWCosRelaxation)


class TestFeasibility(unittest.TestCase):
    def test_univariate_exp(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, aux_var=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('gurobi_direct')
        for xval in [-1, -0.5, 0, 0.5, 1]:
            pval = math.exp(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_pw_exp(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, aux_var=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        m.c.add_partition_point(-0.25)
        m.c.add_partition_point(0.25)
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('gurobi_direct')
        for xval in [-1, -0.5, 0, 0.5, 1]:
            pval = math.exp(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_univariate_log(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(0.5, 1.5))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, aux_var=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONCAVE, f_x_expr=pe.log(m.x))
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('gurobi_direct')
        for xval in [0.5, 0.75, 1, 1.25, 1.5]:
            pval = math.log(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_pw_log(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(0.5, 1.5))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, aux_var=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONCAVE, f_x_expr=pe.log(m.x))
        m.c.add_partition_point(0.9)
        m.c.add_partition_point(1.1)
        m.c.rebuild()
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('gurobi_direct')
        for xval in [0.5, 0.75, 1, 1.25, 1.5]:
            pval = math.log(xval)
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)

    def test_x_fixed(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.x.setlb(0)
        m.x.setub(0)
        m.c = coramin.relaxations.PWUnivariateRelaxation()
        m.c.build(x=m.x, aux_var=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH,
                  shape=coramin.utils.FunctionShape.CONVEX, f_x_expr=pe.exp(m.x))
        m.obj = pe.Objective(expr=m.y)
        opt = pe.SolverFactory('appsi_gurobi')
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 1)
        m.obj.sense = pe.maximize
        res = opt.solve(m)
        self.assertAlmostEqual(m.y.value, 1)

    def test_x_sq(self):
        m = pe.ConcreteModel()
        m.p = pe.Param(initialize=-1, mutable=True)
        m.x = pe.Var(bounds=(-1, 1))
        m.y = pe.Var()
        m.z = pe.Var(bounds=(0, None))
        m.c = coramin.relaxations.PWXSquaredRelaxation()
        m.c.build(x=m.x, aux_var=m.y, relaxation_side=coramin.utils.RelaxationSide.BOTH)
        m.c2 = pe.ConstraintList()
        m.c2.add(m.z >= m.y - m.p)
        m.c2.add(m.z >= m.p - m.y)
        m.obj = pe.Objective(expr=m.z)
        opt = pe.SolverFactory('appsi_gurobi')
        for xval in [-1, -0.5, 0, 0.5, 1]:
            pval = xval**2
            m.x.fix(xval)
            m.p.value = pval
            res = opt.solve(m, tee=False)
            self.assertTrue(res.solver.termination_condition == pe.TerminationCondition.optimal)
            self.assertAlmostEqual(m.y.value, m.p.value, 6)
