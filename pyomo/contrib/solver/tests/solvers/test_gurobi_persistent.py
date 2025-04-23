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
from pyomo.contrib.solver.solvers.gurobi_persistent import GurobiPersistent
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.core.expr.taylor_series import taylor_series_expansion


opt = GurobiPersistent()
if not opt.available():
    raise unittest.SkipTest
import gurobipy


def create_pmedian_model():
    d_dict = {
        (1, 1): 1.777356642700564,
        (1, 2): 1.6698255595592497,
        (1, 3): 1.099139603924817,
        (1, 4): 1.3529705111901453,
        (1, 5): 1.467907742900842,
        (1, 6): 1.5346837414708774,
        (2, 1): 1.9783090609123972,
        (2, 2): 1.130315350158659,
        (2, 3): 1.6712434682302661,
        (2, 4): 1.3642294159473756,
        (2, 5): 1.4888357071619858,
        (2, 6): 1.2030122107340537,
        (3, 1): 1.6661983755713592,
        (3, 2): 1.227663031206932,
        (3, 3): 1.4580640582967632,
        (3, 4): 1.0407223975549575,
        (3, 5): 1.9742897953778287,
        (3, 6): 1.4874760742689066,
        (4, 1): 1.4616138636373597,
        (4, 2): 1.7141471558082002,
        (4, 3): 1.4157281494999725,
        (4, 4): 1.888011688001529,
        (4, 5): 1.0232934487237717,
        (4, 6): 1.8335062677845464,
        (5, 1): 1.468494740997508,
        (5, 2): 1.8114798126442795,
        (5, 3): 1.9455914886158723,
        (5, 4): 1.983088378194899,
        (5, 5): 1.1761820755785306,
        (5, 6): 1.698655759576308,
        (6, 1): 1.108855711312383,
        (6, 2): 1.1602637342062019,
        (6, 3): 1.0928602740245892,
        (6, 4): 1.3140620798928404,
        (6, 5): 1.0165386843386672,
        (6, 6): 1.854049125736362,
        (7, 1): 1.2910160386456968,
        (7, 2): 1.7800475863350327,
        (7, 3): 1.5480965161255695,
        (7, 4): 1.1943306766997612,
        (7, 5): 1.2920382721805297,
        (7, 6): 1.3194527773994338,
        (8, 1): 1.6585982235379078,
        (8, 2): 1.2315210354122292,
        (8, 3): 1.6194303369953538,
        (8, 4): 1.8953386098022103,
        (8, 5): 1.8694342085696831,
        (8, 6): 1.2938069356684523,
        (9, 1): 1.4582048085805495,
        (9, 2): 1.484979797871119,
        (9, 3): 1.2803882693587225,
        (9, 4): 1.3289569463506004,
        (9, 5): 1.9842424240265042,
        (9, 6): 1.0119441379208745,
        (10, 1): 1.1429007682932852,
        (10, 2): 1.6519772165446711,
        (10, 3): 1.0749931799469326,
        (10, 4): 1.2920787022811089,
        (10, 5): 1.7934429721917704,
        (10, 6): 1.9115931008709737,
    }

    model = pyo.ConcreteModel()
    model.N = pyo.Param(initialize=10)
    model.Locations = pyo.RangeSet(1, model.N)
    model.P = pyo.Param(initialize=3)
    model.M = pyo.Param(initialize=6)
    model.Customers = pyo.RangeSet(1, model.M)
    model.d = pyo.Param(
        model.Locations, model.Customers, initialize=d_dict, within=pyo.Reals
    )
    model.x = pyo.Var(model.Locations, model.Customers, bounds=(0.0, 1.0))
    model.y = pyo.Var(model.Locations, within=pyo.Binary)

    def rule(model):
        return sum(
            model.d[n, m] * model.x[n, m]
            for n in model.Locations
            for m in model.Customers
        )

    model.obj = pyo.Objective(rule=rule)

    def rule1(model, m):
        return (sum(model.x[n, m] for n in model.Locations), 1.0)

    model.single_x = pyo.Constraint(model.Customers, rule=rule1)

    def rule2(model, n, m):
        return (None, model.x[n, m] - model.y[n], 0.0)

    model.bound_y = pyo.Constraint(model.Locations, model.Customers, rule=rule2)

    def rule3(model):
        return (sum(model.y[n] for n in model.Locations) - model.P, 0.0)

    model.num_facilities = pyo.Constraint(rule=rule3)

    return model


class TestGurobiPersistentSimpleLPUpdates(unittest.TestCase):
    def setUp(self):
        self.m = pyo.ConcreteModel()
        m = self.m
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.p1 = pyo.Param(mutable=True)
        m.p2 = pyo.Param(mutable=True)
        m.p3 = pyo.Param(mutable=True)
        m.p4 = pyo.Param(mutable=True)
        m.obj = pyo.Objective(expr=m.x + m.y)
        m.c1 = pyo.Constraint(expr=m.y - m.p1 * m.x >= m.p2)
        m.c2 = pyo.Constraint(expr=m.y - m.p3 * m.x >= m.p4)

    def get_solution(self):
        try:
            import numpy as np
        except:
            raise unittest.SkipTest('numpy is not available')
        p1 = self.m.p1.value
        p2 = self.m.p2.value
        p3 = self.m.p3.value
        p4 = self.m.p4.value
        A = np.array([[1, -p1], [1, -p3]])
        rhs = np.array([p2, p4])
        sol = np.linalg.solve(A, rhs)
        x = float(sol[1])
        y = float(sol[0])
        return x, y

    def set_params(self, p1, p2, p3, p4):
        self.m.p1.value = p1
        self.m.p2.value = p2
        self.m.p3.value = p3
        self.m.p4.value = p4

    def test_lp(self):
        self.set_params(-1, -2, 0.1, -2)
        x, y = self.get_solution()
        opt = GurobiPersistent()
        res = opt.solve(self.m)
        self.assertAlmostEqual(x + y, res.incumbent_objective)
        self.assertAlmostEqual(x + y, res.objective_bound)
        self.assertEqual(res.solution_status, SolutionStatus.optimal)
        self.assertTrue(res.incumbent_objective is not None)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)

        self.set_params(-1.25, -1, 0.5, -2)
        opt.config.load_solutions = False
        res = opt.solve(self.m)
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)
        x, y = self.get_solution()
        self.assertNotAlmostEqual(x, self.m.x.value)
        self.assertNotAlmostEqual(y, self.m.y.value)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(x, self.m.x.value)
        self.assertAlmostEqual(y, self.m.y.value)


class TestGurobiPersistent(unittest.TestCase):
    def test_nonconvex_qcp_objective_bound_1(self):
        # the goal of this test is to ensure we can get an objective bound
        # for nonconvex but continuous problems even if a feasible solution
        # is not found
        #
        # This is a fragile test because it could fail if Gurobi's
        # algorithms improve (e.g., a heuristic solution is found before
        # an objective bound of -8 is reached
        #
        # Update: as of Gurobi 11, this test no longer tests the
        # intended behavior (the solver has improved)
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-5, 5))
        m.y = pyo.Var(bounds=(-5, 5))
        m.obj = pyo.Objective(expr=-m.x**2 - m.y)
        m.c1 = pyo.Constraint(expr=m.y <= -2 * m.x + 1)
        m.c2 = pyo.Constraint(expr=m.y <= m.x - 2)
        opt = GurobiPersistent()
        opt.config.solver_options['nonconvex'] = 2
        opt.config.solver_options['BestBdStop'] = -8
        opt.config.load_solutions = False
        opt.config.raise_exception_on_nonoptimal_result = False
        res = opt.solve(m)
        if opt.version() < (11, 0):
            self.assertEqual(res.incumbent_objective, None)
        else:
            self.assertEqual(res.incumbent_objective, -4)
        self.assertAlmostEqual(res.objective_bound, -8)

    def test_nonconvex_qcp_objective_bound_2(self):
        # the goal of this test is to ensure we can objective_bound
        # properly for nonconvex but continuous problems when the solver
        # terminates with a nonzero gap
        #
        # This is a fragile test because it could fail if Gurobi's
        # algorithms change
        #
        # Update: as of Gurobi 11, this test no longer tests the
        # intended behavior (the solver has improved)
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-5, 5))
        m.y = pyo.Var(bounds=(-5, 5))
        m.obj = pyo.Objective(expr=-m.x**2 - m.y)
        m.c1 = pyo.Constraint(expr=m.y <= -2 * m.x + 1)
        m.c2 = pyo.Constraint(expr=m.y <= m.x - 2)
        opt = GurobiPersistent()
        opt.config.solver_options['nonconvex'] = 2
        opt.config.solver_options['MIPGap'] = 0.5
        res = opt.solve(m)
        self.assertAlmostEqual(res.incumbent_objective, -4)
        if opt.version() < (11, 0):
            self.assertAlmostEqual(res.objective_bound, -6)
        else:
            self.assertAlmostEqual(res.objective_bound, -4)

    def test_range_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.xl = pyo.Param(initialize=-1, mutable=True)
        m.xu = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Constraint(expr=pyo.inequality(m.xl, m.x, m.xu))
        m.obj = pyo.Objective(expr=m.x)

        opt = GurobiPersistent()
        opt.set_instance(m)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.xl.value = -3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.xu.value = 3
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_quadratic_constraint_with_params(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y >= m.a * m.x**2 + m.b * m.x + m.c)

        opt = GurobiPersistent()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(
            m.y.value, m.a.value * m.x.value**2 + m.b.value * m.x.value + m.c.value
        )

        m.a.value = 2
        m.b.value = 4
        m.c.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(
            m.y.value, m.a.value * m.x.value**2 + m.b.value * m.x.value + m.c.value
        )

    def test_quadratic_objective(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.x = pyo.Var()
        m.obj = pyo.Objective(expr=m.a * m.x**2 + m.b * m.x + m.c)

        opt = GurobiPersistent()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(
            res.incumbent_objective,
            m.a.value * m.x.value**2 + m.b.value * m.x.value + m.c.value,
        )

        m.a.value = 2
        m.b.value = 4
        m.c.value = -1
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(
            res.incumbent_objective,
            m.a.value * m.x.value**2 + m.b.value * m.x.value + m.c.value,
        )

    def test_var_bounds(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-1, 1))
        m.obj = pyo.Objective(expr=m.x)

        opt = GurobiPersistent()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -1)

        m.x.setlb(-3)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -3)

        del m.obj
        m.obj = pyo.Objective(expr=m.x, sense=pyo.maximize)

        opt = GurobiPersistent()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)

        m.x.setub(3)
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 3)

    def test_fixed_var(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Param(initialize=1, mutable=True)
        m.b = pyo.Param(initialize=1, mutable=True)
        m.c = pyo.Param(initialize=1, mutable=True)
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.y)
        m.con = pyo.Constraint(expr=m.y >= m.a * m.x**2 + m.b * m.x + m.c)

        m.x.fix(1)
        opt = GurobiPersistent()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 3)

        m.x.value = 2
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, 2)
        self.assertAlmostEqual(m.y.value, 7)

        m.x.unfix()
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -m.b.value / (2 * m.a.value))
        self.assertAlmostEqual(
            m.y.value, m.a.value * m.x.value**2 + m.b.value * m.x.value + m.c.value
        )

    def test_linear_constraint_attr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.x + m.y == 1)

        opt = GurobiPersistent()
        opt.set_instance(m)
        opt.set_linear_constraint_attr(m.c, 'Lazy', 1)
        self.assertEqual(opt.get_linear_constraint_attr(m.c, 'Lazy'), 1)

    def test_quadratic_constraint_attr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.c = pyo.Constraint(expr=m.y >= m.x**2)

        opt = GurobiPersistent()
        opt.set_instance(m)
        self.assertEqual(opt.get_quadratic_constraint_attr(m.c, 'QCRHS'), 0)

    def test_var_attr(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.x)

        opt = GurobiPersistent()
        opt.set_instance(m)
        opt.set_var_attr(m.x, 'Start', 1)
        self.assertEqual(opt.get_var_attr(m.x, 'Start'), 1)

    def test_callback(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 4))
        m.y = pyo.Var(within=pyo.Integers, bounds=(0, None))
        m.obj = pyo.Objective(expr=2 * m.x + m.y)
        m.cons = pyo.ConstraintList()

        def _add_cut(xval):
            m.x.value = xval
            return m.cons.add(m.y >= taylor_series_expansion((m.x - 2) ** 2))

        _add_cut(0)
        _add_cut(4)

        opt = GurobiPersistent()
        opt.set_instance(m)
        opt.set_gurobi_param('PreCrush', 1)
        opt.set_gurobi_param('LazyConstraints', 1)

        def _my_callback(cb_m, cb_opt, cb_where):
            if cb_where == gurobipy.GRB.Callback.MIPSOL:
                cb_opt.cbGetSolution(variables=[m.x, m.y])
                if m.y.value < (m.x.value - 2) ** 2 - 1e-6:
                    cb_opt.cbLazy(_add_cut(m.x.value))

        opt.set_callback(_my_callback)
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, 1)
        self.assertAlmostEqual(m.y.value, 1)

    def test_nonconvex(self):
        if gurobipy.GRB.VERSION_MAJOR < 9:
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c = pyo.Constraint(expr=m.y == (m.x - 1) ** 2 - 2)
        opt = GurobiPersistent()
        opt.config.solver_options['nonconvex'] = 2
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.3660254037844423, 2)
        self.assertAlmostEqual(m.y.value, -0.13397459621555508, 2)

    def test_nonconvex2(self):
        if gurobipy.GRB.VERSION_MAJOR < 9:
            raise unittest.SkipTest
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=0 <= -m.y + (m.x - 1) ** 2 - 2)
        m.c2 = pyo.Constraint(expr=0 >= -m.y + (m.x - 1) ** 2 - 2)
        opt = GurobiPersistent()
        opt.config.solver_options['nonconvex'] = 2
        opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.3660254037844423, 2)
        self.assertAlmostEqual(m.y.value, -0.13397459621555508, 2)

    def test_solution_number(self):
        m = create_pmedian_model()
        opt = GurobiPersistent()
        opt.config.solver_options['PoolSolutions'] = 3
        opt.config.solver_options['PoolSearchMode'] = 2
        res = opt.solve(m)
        num_solutions = opt.get_model_attr('SolCount')
        self.assertEqual(num_solutions, 3)
        res.solution_loader.load_vars(solution_number=0)
        self.assertAlmostEqual(pyo.value(m.obj.expr), 6.431184939357673)
        res.solution_loader.load_vars(solution_number=1)
        self.assertAlmostEqual(pyo.value(m.obj.expr), 6.584793218502477)
        res.solution_loader.load_vars(solution_number=2)
        self.assertAlmostEqual(pyo.value(m.obj.expr), 6.592304628123309)

    def test_zero_time_limit(self):
        m = create_pmedian_model()
        opt = GurobiPersistent()
        opt.config.time_limit = 0
        opt.config.load_solutions = False
        opt.config.raise_exception_on_nonoptimal_result = False
        res = opt.solve(m)
        num_solutions = opt.get_model_attr('SolCount')

        # Behavior is different on different platforms, so
        # we have to see if there are any solutions
        # This means that there is no guarantee we are testing
        # what we are trying to test. Unfortunately, I'm
        # not sure of a good way to guarantee that
        if num_solutions == 0:
            self.assertIsNone(res.incumbent_objective)


class TestManualModel(unittest.TestCase):
    def setUp(self):
        opt = GurobiPersistent()
        opt.config.auto_updates.check_for_new_or_removed_params = False
        opt.config.auto_updates.check_for_new_or_removed_vars = False
        opt.config.auto_updates.check_for_new_or_removed_constraints = False
        opt.config.auto_updates.update_parameters = False
        opt.config.auto_updates.update_vars = False
        opt.config.auto_updates.update_constraints = False
        opt.config.auto_updates.update_named_expressions = False
        self.opt = opt

    def test_basics(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(-10, 10))
        m.y = pyo.Var()
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pyo.Constraint(expr=m.y >= 2 * m.x + 1)

        opt = self.opt
        opt.set_instance(m)

        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -10)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 10)

        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -0.4)

        m.c2 = pyo.Constraint(expr=m.y >= -m.x + 1)
        opt.add_constraints([m.c2])
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 2)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        opt.config.load_solutions = False
        res = opt.solve(m)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)
        res.solution_loader.load_vars()
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

        opt.remove_constraints([m.c2])
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-6)
        opt.config.solver_options['FeasibilityTol'] = 1e-7
        opt.config.load_solutions = True
        res = opt.solve(m)
        self.assertEqual(opt.get_gurobi_param_info('FeasibilityTol')[2], 1e-7)
        self.assertAlmostEqual(m.x.value, -0.4)
        self.assertAlmostEqual(m.y.value, 0.2)

        m.x.setlb(-5)
        m.x.setub(5)
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.x.fix(0)
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), 0)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 0)

        m.x.unfix()
        opt.update_variables([m.x])
        self.assertEqual(opt.get_var_attr(m.x, 'LB'), -5)
        self.assertEqual(opt.get_var_attr(m.x, 'UB'), 5)

        m.c2 = pyo.Constraint(expr=m.y >= m.x**2)
        opt.add_constraints([m.c2])
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 1)

        opt.remove_constraints([m.c2])
        m.del_component(m.c2)
        self.assertEqual(opt.get_model_attr('NumVars'), 2)
        self.assertEqual(opt.get_model_attr('NumConstrs'), 1)
        self.assertEqual(opt.get_model_attr('NumQConstrs'), 0)

        m.z = pyo.Var()
        opt.add_variables([m.z])
        self.assertEqual(opt.get_model_attr('NumVars'), 3)
        opt.remove_variables([m.z])
        del m.z
        self.assertEqual(opt.get_model_attr('NumVars'), 2)

    def test_update1(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

        opt.remove_constraints([m.c1])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)

        opt.add_constraints([m.c1])
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update2(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c2 = pyo.Constraint(expr=m.x + m.y == 1)

        opt = self.opt
        opt.config.symbolic_solver_labels = True
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)

        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update3(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x**2 + m.y**2)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        m.c2 = pyo.Constraint(expr=m.y >= m.x**2)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumQConstrs'), 1)

    def test_update4(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var()
        m.z = pyo.Var()
        m.obj = pyo.Objective(expr=m.z)
        m.c1 = pyo.Constraint(expr=m.z >= m.x + m.y)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        m.c2 = pyo.Constraint(expr=m.y >= m.x)
        opt.add_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)
        opt.remove_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumConstrs'), 1)

    def test_update5(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        opt = self.opt
        opt.set_instance(m)
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

        opt.remove_sos_constraints([m.c1])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)

        opt.add_sos_constraints([m.c1])
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 0)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)

    def test_update6(self):
        m = pyo.ConcreteModel()
        m.a = pyo.Set(initialize=[1, 2, 3], ordered=True)
        m.x = pyo.Var(m.a, within=pyo.Binary)
        m.y = pyo.Var(within=pyo.Binary)
        m.obj = pyo.Objective(expr=m.y)
        m.c1 = pyo.SOSConstraint(var=m.x, sos=1)

        opt = self.opt
        opt.set_instance(m)
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        m.c2 = pyo.SOSConstraint(var=m.x, sos=2)
        opt.add_sos_constraints([m.c2])
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
        opt.remove_sos_constraints([m.c2])
        opt.update()
        self.assertEqual(opt._solver_model.getAttr('NumSOS'), 1)
