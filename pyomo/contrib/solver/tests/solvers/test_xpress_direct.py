# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import math
import os
import tempfile
import types

import pyomo.environ as pyo
import pyomo.common.unittest as unittest
from pyomo.common.timing import HierarchicalTimer

from pyomo.contrib.solver.common.results import TerminationCondition
from pyomo.contrib.solver.common.util import (
    IncompatibleModelError,
    NoDualsError,
    NoFeasibleSolutionError,
    NoReducedCostsError,
    NoSolutionError,
)
from pyomo.contrib.solver.common.results import SolutionStatus
from pyomo.contrib.solver.solvers.xpress import XpressDirect
from pyomo.contrib.solver.solvers.xpress.xpress_base import _exit_external_function, xp
from pyomo.contrib.solver.tests.solvers._xpress_test_utils import (
    _simple_lp,
    _simple_mip,
    _solve_and_check,
    _solve_lp_no_load,
)

if not XpressDirect().available():
    raise unittest.SkipTest('Xpress not available')


def _infeasible():
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.c1 = pyo.Constraint(expr=m.x >= 10)
    m.c2 = pyo.Constraint(expr=m.x <= 1)
    m.obj = pyo.Objective(expr=m.x)
    return m


@unittest.pytest.mark.solver("xpress_direct")
class TestXpressDirect(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_symbolic_solver_labels_lp(self):
        m = pyo.ConcreteModel()
        m.distinctive_x = pyo.Var(domain=pyo.NonNegativeReals)
        m.distinctive_y = pyo.Var(domain=pyo.NonNegativeReals)
        m.distinctive_c1 = pyo.Constraint(expr=m.distinctive_x + m.distinctive_y <= 4)
        m.distinctive_c2 = pyo.Constraint(
            expr=2 * m.distinctive_x + m.distinctive_y <= 6
        )
        m.obj = pyo.Objective(expr=-m.distinctive_x - 2 * m.distinctive_y)
        res = _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -8.0,
                'vars': [(m.distinctive_x, 0.0), (m.distinctive_y, 4.0)],
            },
            symbolic_solver_labels=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'm')
            res.solution_loader._xp_prob.writeProb(base + '.lp', flags='l')
            with open(base + '.lp', 'r') as f:
                content = f.read()
        self.assertIn('distinctive_x', content)
        self.assertIn('distinctive_c1', content)

    def test_symbolic_solver_labels_mip(self):
        m = pyo.ConcreteModel()
        m.distinctive_x = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.distinctive_y = pyo.Var(domain=pyo.NonNegativeIntegers)
        m.distinctive_c1 = pyo.Constraint(expr=m.distinctive_x + m.distinctive_y <= 4)
        m.obj = pyo.Objective(expr=-m.distinctive_x - 2 * m.distinctive_y)
        res = _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -8.0,
                'vars': [(m.distinctive_x, 0.0), (m.distinctive_y, 4.0)],
            },
            symbolic_solver_labels=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, 'm')
            res.solution_loader._xp_prob.writeProb(base + '.lp', flags='l')
            with open(base + '.lp', 'r') as f:
                content = f.read()
        self.assertIn('distinctive_x', content)
        self.assertIn('distinctive_c1', content)

    def test_positive_time_limit(self):
        m = _simple_lp()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            time_limit=60,
        )

    def test_solver_options_passthrough(self):
        m = _simple_lp()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            solver_options={'outputlog': 0},
        )
        with self.assertRaises(Exception):
            self.opt.solve(m, solver_options={'_invalid_control_xyz': 1})

    def test_rel_gap(self):
        m = _simple_mip()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            rel_gap=0.01,
        )

    def test_threads(self):
        m = _simple_lp()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            threads=1,
        )

    def test_mip_no_duals_no_reduced_costs(self):
        m = _simple_mip()
        res = self.opt.solve(m, load_solutions=False)
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        with self.assertRaises(NoDualsError):
            res.solution_loader.get_duals()
        with self.assertRaises(NoReducedCostsError):
            res.solution_loader.get_reduced_costs()

    def test_get_vars_no_solution(self):
        m = _infeasible()
        res = _solve_and_check(
            self,
            self.opt,
            m,
            {
                'termination': TerminationCondition.provenInfeasible,
                'status': SolutionStatus.infeasible,
            },
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )
        with self.assertRaises(NoSolutionError):
            res.solution_loader.get_vars()

    def test_warmstart(self):
        m = _simple_mip()
        m.x.set_value(0)
        m.y.set_value(4)
        res = _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        self.assertGreaterEqual(res.extra_info.mip_solutions_found, 1)

    def test_extra_info_and_timing(self):
        m = _simple_lp()
        res = _solve_and_check(
            self, self.opt, m, {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]}
        )
        self.assertGreaterEqual(res.timing_info.xpress_time, 0)
        self.assertGreaterEqual(res.extra_info.simplex_iterations, 1)
        self.assertGreaterEqual(res.extra_info.barrier_iterations, 0)
        self.assertEqual(res.extra_info.node_count, 0)
        self.assertEqual(res.extra_info.mip_solutions_found, 0)

    def test_load_solutions_infeasible(self):
        m = _infeasible()
        with self.assertRaises(NoFeasibleSolutionError):
            self.opt.solve(
                m, raise_exception_on_nonoptimal_result=False, load_solutions=True
            )

    def test_load_vars_subset(self):
        m, res = _solve_lp_no_load(self.opt)
        m.x.set_value(99.0)
        m.y.set_value(99.0)
        res.solution_loader.load_vars([m.y])
        self.assertAlmostEqual(m.y.value, 4.0)
        self.assertAlmostEqual(m.x.value, 99.0)
        res.solution_loader.load_vars([m.x, m.y])
        self.assertAlmostEqual(m.x.value, 0.0)
        self.assertAlmostEqual(m.y.value, 4.0)

    def test_get_vars_subset(self):
        m, res = _solve_lp_no_load(self.opt)
        result = res.solution_loader.get_vars([m.x])
        self.assertIn(m.x, result)
        self.assertNotIn(m.y, result)
        self.assertAlmostEqual(result[m.x], 0.0)
        result = res.solution_loader.get_vars([m.x, m.y])
        self.assertAlmostEqual(result[m.x], 0.0)
        self.assertAlmostEqual(result[m.y], 4.0)
        result = res.solution_loader.get_vars([m.y, m.x])
        self.assertAlmostEqual(result[m.x], 0.0)
        self.assertAlmostEqual(result[m.y], 4.0)

    def test_get_reduced_costs_subset(self):
        m, res = _solve_lp_no_load(self.opt)
        result = res.solution_loader.get_reduced_costs([m.y])
        self.assertIn(m.y, result)
        self.assertNotIn(m.x, result)
        self.assertAlmostEqual(result[m.y], 0.0)
        result = res.solution_loader.get_reduced_costs([m.x, m.y])
        self.assertIn(m.x, result)
        self.assertIn(m.y, result)
        self.assertAlmostEqual(result[m.y], 0.0)

    def test_get_vars_all(self):
        m, res = _solve_lp_no_load(self.opt)
        result = res.solution_loader.get_vars()
        self.assertIn(m.x, result)
        self.assertIn(m.y, result)
        self.assertAlmostEqual(result[m.x], 0.0)
        self.assertAlmostEqual(result[m.y], 4.0)

    def test_multiple_objectives_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj1 = pyo.Objective(expr=m.x)
        m.obj2 = pyo.Objective(expr=-m.x)
        with self.assertRaises(IncompatibleModelError):
            self.opt.solve(m)

    def test_sos1_direct(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 3.0, 'vars': [(m.x[1], 0.0), (m.x[2], 0.0), (m.x[3], 1.0)]},
        )

    def test_sos1_vars_not_in_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.y = pyo.Var(bounds=(0, 10))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.y)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': 0.0,
                'vars': [(m.x[1], 0.0), (m.x[2], 0.0), (m.x[3], 0.0), (m.y, 0.0)],
            },
        )

    def test_sos1_no_duplicate_columns(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], domain=pyo.NonNegativeReals, bounds=(0, 1))
        m.sos1 = pyo.SOSConstraint(var=m.x, sos=1, weights={1: 1.0, 2: 2.0, 3: 3.0})
        m.obj = pyo.Objective(expr=m.x[1] + 2 * m.x[2] + 3 * m.x[3], sense=pyo.maximize)
        xp_prob = self.opt._create_xpress_model(
            m,
            self.opt.config,
            __import__(
                'pyomo.common.timing', fromlist=['HierarchicalTimer']
            ).HierarchicalTimer(),
        )[0]
        self.assertEqual(xp_prob.attributes.cols, 3)

    def test_get_duals_single_constraint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.c = pyo.Constraint(expr=m.x >= 2)
        m.obj = pyo.Objective(expr=m.x)
        res = _solve_and_check(
            self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]}
        )
        duals = res.solution_loader.get_duals([m.c])
        self.assertIn(m.c, duals)
        self.assertIsInstance(duals[m.c], float)

    def test_reduced_costs_value_correctness(self):
        m, res = _solve_lp_no_load(self.opt)
        rcs = res.solution_loader.get_reduced_costs()
        self.assertAlmostEqual(rcs[m.x], 1.0, places=6)
        self.assertAlmostEqual(rcs[m.y], 0.0, places=6)

    def test_duals_value_correctness(self):
        m, res = _solve_lp_no_load(self.opt)
        duals = res.solution_loader.get_duals()
        self.assertAlmostEqual(duals[m.c1], -2.0, places=6)
        self.assertAlmostEqual(duals[m.c2], 0.0, places=6)


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressDirectQuadratic(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_qp_objective_direct(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=m.x + m.y >= 1)
        m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.5, 'vars': [(m.x, 0.5), (m.y, 0.5)]}
        )

    def test_qcp_constraint_direct(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, None))
        m.y = pyo.Var(bounds=(0, None))
        m.qc = pyo.Constraint(expr=m.x**2 + m.y**2 <= 1)
        m.obj = pyo.Objective(expr=-(m.x + m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'objective': -math.sqrt(2),
                'vars': [(m.x, math.sqrt(2) / 2), (m.y, math.sqrt(2) / 2)],
                'obj_places': 5,
                'var_places': 5,
            },
        )

    def test_nl_cubic_constraint_direct(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x**3 >= 1)
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(self, self.opt, m, {'objective': 1.0, 'vars': [(m.x, 1.0)]})


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressDirectMisc(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_nl_cubic_objective_direct(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x**3)
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})

    def test_abs_gap_passthrough(self):
        m = _simple_mip()
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': -8.0, 'vars': [(m.x, 0.0), (m.y, 4.0)]},
            abs_gap=0.5,
        )

    def test_working_dir_chdir_and_restore(self):
        m = _simple_lp()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            self.opt.solve(m, working_dir=tmp)
            self.assertEqual(os.getcwd(), original_cwd)

    def test_empty_constraint_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(2, 10))
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]})

    def test_no_objective_feasibility(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x >= 3)
        res = self.opt.solve(m)
        self.assertEqual(
            res.termination_condition, TerminationCondition.convergenceCriteriaSatisfied
        )
        self.assertIsNone(res.incumbent_objective)

    def test_constant_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=5.0)
        _solve_and_check(self, self.opt, m, {'objective': 5.0, 'vars': [(m.x, 1.0)]})

    def test_range_constraint_lp(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(domain=pyo.NonNegativeReals)
        m.y = pyo.Var(domain=pyo.NonNegativeReals)
        m.c = pyo.Constraint(expr=pyo.inequality(1, m.x + m.y, 3))
        m.obj = pyo.Objective(expr=-2 * m.x - m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': -6.0, 'vars': [(m.x, 3.0), (m.y, 0.0)]}
        )

    def test_get_duals_no_args(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 10))
        m.c1 = pyo.Constraint(expr=m.x >= 1)
        m.c2 = pyo.Constraint(expr=m.x >= 2)
        m.obj = pyo.Objective(expr=m.x)
        res = _solve_and_check(
            self, self.opt, m, {'objective': 2.0, 'vars': [(m.x, 2.0)]}
        )
        duals = res.solution_loader.get_duals()
        self.assertIn(m.c1, duals)
        self.assertIn(m.c2, duals)

    def test_fixed_var_without_value_raises(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.x.fix()
        m.obj = pyo.Objective(expr=m.x)
        self.assertIsNone(m.x.value)
        with self.assertRaises(ValueError):
            self.opt.solve(m)

    def test_controls_unit(self):
        m = _simple_lp()
        opt = XpressDirect()
        xp_prob, _, _ = opt._create_xpress_model(m, opt.config, HierarchicalTimer())
        config = opt.config({'time_limit': 42, 'threads': 2})
        opt._apply_solver_controls(xp_prob, config)
        self.assertEqual(xp_prob.controls.timelimit, 42.0)
        self.assertEqual(xp_prob.controls.threads, 2)

    def test_infeasible_model_returns_infeasible_result(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(5, 3))
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'termination': TerminationCondition.provenInfeasible,
                'status': SolutionStatus.infeasible,
            },
            raise_exception_on_nonoptimal_result=False,
            load_solutions=False,
        )

    def test_time_limit_zero(self):
        m = _simple_lp()
        opt = XpressDirect()
        xp_prob, _, _ = opt._create_xpress_model(m, opt.config, HierarchicalTimer())
        config = opt.config({'time_limit': 0})
        opt._apply_solver_controls(xp_prob, config)
        self.assertEqual(xp_prob.controls.timelimit, 0.0)

    def test_working_dir_restored_on_exception(self):
        m = _simple_lp()
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                self.opt.solve(
                    m, working_dir=tmp, solver_options={'_invalid_control_xyz': 1}
                )
            except Exception:
                pass
            self.assertEqual(os.getcwd(), original_cwd)


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressDirectNLP(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_nl_exp_objective_linear_constraints(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 3))
        m.c = pyo.Constraint(expr=m.x >= 1)
        m.obj = pyo.Objective(expr=pyo.exp(m.x))
        _solve_and_check(self, self.opt, m, {'objective': math.e, 'vars': [(m.x, 1.0)]})

    def test_nl_sin_constraints_linear_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi / 2))
        m.y = pyo.Var(bounds=(0, 1))
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.y <= 1)
        m.obj = pyo.Objective(expr=m.x + m.y)
        _solve_and_check(
            self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0), (m.y, 0.0)]}
        )

    def test_nl_objective_nl_constraint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 2))
        m.c = pyo.Constraint(expr=pyo.exp(m.x) <= 2)
        m.obj = pyo.Objective(expr=pyo.sin(m.x))
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 0.0)]})

    def test_nl_range_constraint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.c = pyo.Constraint(expr=pyo.inequality(0.5, pyo.sin(m.x), 1.0))
        m.obj = pyo.Objective(expr=m.x)
        _solve_and_check(
            self, self.opt, m, {'objective': math.pi / 6, 'vars': [(m.x, math.pi / 6)]}
        )
        self.assertAlmostEqual(pyo.sin(pyo.value(m.x)), 0.5, places=6)

    def test_fixed_variable_in_nl_constraint(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, math.pi))
        m.y = pyo.Var(bounds=(0, 10))
        m.x.fix(math.pi / 2)
        m.c = pyo.Constraint(expr=pyo.sin(m.x) + m.y <= 5)
        m.obj = pyo.Objective(expr=m.y)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'objective': 0.0, 'vars': [(m.x, math.pi / 2), (m.y, 0.0)]},
        )

    def test_nl_abs_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.obj = pyo.Objective(expr=abs(m.x - 2))
        _solve_and_check(self, self.opt, m, {'objective': 0.0, 'vars': [(m.x, 2.0)]})


@unittest.pytest.mark.solver('xpress_direct')
class TestXpressExternalFunction(unittest.TestCase):
    def setUp(self):
        self.opt = XpressDirect()

    def test_external_function_no_gradient(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(1, 5))
        m.f = pyo.ExternalFunction(function=lambda x, y: x**2 + y)
        m.obj = pyo.Objective(expr=m.f(m.x, m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 1.0)],
            },
        )

    def test_external_function_with_gradient(self):
        def f(x, y):
            return x**2 + y

        def grad(args, fixed):
            x, _ = args
            return [2 * x, 1.0]

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(1, 5))
        m.f = pyo.ExternalFunction(function=f, gradient=grad)
        m.obj = pyo.Objective(expr=m.f(m.x, m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 1.0)],
            },
        )

    def test_non_supported_external_function_raises(self):
        node = types.SimpleNamespace(_fcn=object())
        with self.assertRaises(IncompatibleModelError):
            _exit_external_function(None, node)

    def test_external_function_in_constraint(self):
        def g(y):
            return y

        def grad(args, fixed):
            return [1.0]

        m = pyo.ConcreteModel()
        m.y = pyo.Var(bounds=(0, 5))
        m.g = pyo.ExternalFunction(function=g, gradient=grad)
        m.c = pyo.Constraint(expr=m.g(m.y) >= 1)
        m.obj = pyo.Objective(expr=m.y)
        _solve_and_check(
            self,
            self.opt,
            m,
            {'status': SolutionStatus.feasible, 'objective': 1.0, 'vars': [(m.y, 1.0)]},
        )

    def test_external_function_multiple(self):
        def f1(x):
            return x**2

        def f2(y):
            return y + 1.0

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(0, 5))
        m.f1 = pyo.ExternalFunction(function=f1)
        m.f2 = pyo.ExternalFunction(function=f2)
        m.obj = pyo.Objective(expr=m.f1(m.x) + m.f2(m.y))
        if xp.featurequery("Community"):  # Free community license
            with self.assertRaisesRegex(Exception, r"^\?1152"):
                self.opt.solve(m)
        else:
            _solve_and_check(
                self,
                self.opt,
                m,
                {
                    'status': SolutionStatus.feasible,
                    'objective': 1.0,
                    'vars': [(m.x, 0.0), (m.y, 0.0)],
                },
            )

    def test_external_function_fgh_callback(self):
        def fgh_func(args, fgh_flag, fixed):
            x, y = args
            f = x**2 + y
            g = [2.0 * x, 1.0] if fgh_flag else None
            return f, g, None

        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 5))
        m.y = pyo.Var(bounds=(1, 5))
        m.f = pyo.ExternalFunction(fgh=fgh_func)
        m.obj = pyo.Objective(expr=m.f(m.x, m.y))
        _solve_and_check(
            self,
            self.opt,
            m,
            {
                'status': SolutionStatus.feasible,
                'objective': 1.0,
                'vars': [(m.x, 0.0), (m.y, 1.0)],
            },
        )


if __name__ == '__main__':
    unittest.main()
