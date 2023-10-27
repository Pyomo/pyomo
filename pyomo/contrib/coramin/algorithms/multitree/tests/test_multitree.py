import math

import coramin
from coramin.third_party.minlplib_tools import get_minlplib, get_minlplib_instancedata
import unittest
from pyomo.contrib import appsi
import os
import logging
from suspect.pyomo.osil_reader import read_osil
import math
from pyomo.common import download
import pyomo.environ as pe
from pyomo.core.base.block import _BlockData


def _get_sol(pname):
    current_dir = os.getcwd()
    target_fname = os.path.join(current_dir, f'{pname}.sol')
    downloader = download.FileDownloader()
    downloader.set_destination_filename(target_fname)
    downloader.get_binary_file(f'http://www.minlplib.org/sol/{pname}.p1.sol')
    res = dict()
    f = open(target_fname, 'r')
    for line in f.readlines():
        l = line.split()
        vname = l[0]
        vval = float(l[1])
        if vname != 'objvar':
            res[vname] = vval
    f.close()
    return res


class Helper(unittest.TestCase):
    def _check_relative_diff(self, expected, got, abs_tol=1e-3, rel_tol=1e-3):
        abs_diff = abs(expected - got)
        if expected == 0:
            rel_diff = math.inf
        else:
            rel_diff = abs_diff / abs(expected)
        success = abs_diff <= abs_tol or rel_diff <= rel_tol
        self.assertTrue(success, msg=f'\n    expected: {expected}\n    got: {got}\n    abs diff: {abs_diff}\n    rel diff: {rel_diff}')


class TestMultiTreeWithMINLPLib(Helper):
    @classmethod
    def setUpClass(self) -> None:
        self.test_problems = {'batch': 285506.5082,
                              'ball_mk3_10': None,
                              'ball_mk2_10': 0,
                              'syn05m': 837.73240090,
                              'autocorr_bern20-03': -72,
                              'chem': -47.70651483,
                              'alkyl': -1.76499965}
        self.primal_sol = dict()
        self.primal_sol['batch'] = _get_sol('batch')
        self.primal_sol['alkyl'] = _get_sol('alkyl')
        self.primal_sol['ball_mk2_10'] = _get_sol('ball_mk2_10')
        self.primal_sol['syn05m'] = _get_sol('syn05m')
        self.primal_sol['autocorr_bern20-03'] = _get_sol('autocorr_bern20-03')
        self.primal_sol['chem'] = _get_sol('chem')
        for pname in self.test_problems.keys():
            get_minlplib(problem_name=pname)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        nlp_solver.config.log_level = logging.DEBUG
        self.opt = coramin.algorithms.MultiTree(mip_solver=mip_solver,
                                                nlp_solver=nlp_solver)

    @classmethod
    def tearDownClass(self) -> None:
        current_dir = os.getcwd()
        for pname in self.test_problems.keys():
            os.remove(os.path.join(current_dir, 'minlplib', 'osil', f'{pname}.osil'))
        os.rmdir(os.path.join(current_dir, 'minlplib', 'osil'))
        os.rmdir(os.path.join(current_dir, 'minlplib'))
        for pname in self.primal_sol.keys():
            os.remove(os.path.join(current_dir, f'{pname}.sol'))

    def get_model(self, pname):
        current_dir = os.getcwd()
        fname = os.path.join(current_dir, 'minlplib', 'osil', f'{pname}.osil')
        m = read_osil(fname, objective_prefix='obj_')
        return m

    def _check_primal_sol(self, pname, m: _BlockData, res: appsi.base.Results):
        expected_by_str = self.primal_sol[pname]
        expected_by_var = pe.ComponentMap()
        for vname, vval in expected_by_str.items():
            v = getattr(m, vname)
            expected_by_var[v] = vval
        got = res.solution_loader.get_primals()
        for v, val in expected_by_var.items():
            self._check_relative_diff(val, got[v])
        got = res.solution_loader.get_primals(vars_to_load=list(expected_by_var.keys()))
        for v, val in expected_by_var.items():
            self._check_relative_diff(val, got[v])

    def optimal_helper(self, pname, check_primal_sol=True):
        m = self.get_model(pname)
        res = self.opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self._check_relative_diff(self.test_problems[pname], res.best_feasible_objective,
                                  abs_tol=self.opt.config.abs_gap, rel_tol=self.opt.config.mip_gap)
        self._check_relative_diff(self.test_problems[pname], res.best_objective_bound,
                                  abs_tol=self.opt.config.abs_gap, rel_tol=self.opt.config.mip_gap)
        if check_primal_sol:
            self._check_primal_sol(pname, m, res)

    def infeasible_helper(self, pname):
        m = self.get_model(pname)
        self.opt.config.load_solution = False
        res = self.opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.infeasible)
        self.opt.config.load_solution = True

    def time_limit_helper(self, pname):
        orig_time_limit = self.opt.config.time_limit
        self.opt.config.load_solution = False
        for new_limit in [0, 0.1, 0.2, 0.3]:
            self.opt.config.time_limit = new_limit
            m = self.get_model(pname)
            res = self.opt.solve(m)
            self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.maxTimeLimit)
        self.opt.config.load_solution = True

    def test_batch(self):
        self.optimal_helper('batch')

    def test_ball_mk2_10(self):
        self.optimal_helper('ball_mk2_10')

    def test_alkyl(self):
        self.optimal_helper('alkyl')

    def test_syn05m(self):
        self.optimal_helper('syn05m')

    def test_autocorr_bern20_03(self):
        self.optimal_helper('autocorr_bern20-03', check_primal_sol=False)

    def test_chem(self):
        orig_config = self.opt.config()
        self.opt.config.root_obbt_max_iter = 10
        self.opt.config.mip_gap = 0.05
        self.optimal_helper('chem')
        self.opt.config = orig_config

    def test_time_limit(self):
        self.time_limit_helper('alkyl')

    def test_ball_mk3_10(self):
        self.infeasible_helper('ball_mk3_10')

    def test_available(self):
        avail = self.opt.available()
        assert avail in appsi.base.Solver.Availability


class TestMultiTree(Helper):
    def test_convex_overestimator(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=(m.x + 1)**2 - 0.2*m.y)
        m.c = pe.Constraint(expr=m.y <= m.x**2)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        nlp_solver.config.log_level = logging.DEBUG
        opt = coramin.algorithms.MultiTree(mip_solver=mip_solver,
                                           nlp_solver=nlp_solver)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self._check_relative_diff(-0.25, res.best_feasible_objective,
                             abs_tol=opt.config.abs_gap, rel_tol=opt.config.mip_gap)
        self._check_relative_diff(-0.25, res.best_objective_bound,
                             abs_tol=opt.config.abs_gap, rel_tol=opt.config.mip_gap)
        self._check_relative_diff(-1.250953, m.x.value, 1e-2, 1e-2)
        self._check_relative_diff(1.5648825, m.y.value, 1e-2, 1e-2)

    def test_max_iter(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var()
        m.obj = pe.Objective(expr=(m.x + 1)**2 - 0.2*m.y)
        m.c = pe.Constraint(expr=m.y <= m.x**2)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        nlp_solver.config.log_level = logging.DEBUG
        opt = coramin.algorithms.MultiTree(mip_solver=mip_solver,
                                           nlp_solver=nlp_solver)
        opt.config.max_iter = 3
        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.maxIterations)
        self.assertIsNone(res.best_feasible_objective)
        opt.config.max_iter = 10
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.maxIterations)
        self.assertIsNotNone(res.best_feasible_objective)

    def test_nlp_infeas_fbbt(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1), domain=pe.Integers)
        m.y = pe.Var(domain=pe.Integers)
        m.obj = pe.Objective(expr=(m.x + 1)**2 - 0.2*m.y)
        m.c1 = pe.Constraint(expr=m.y <= (m.x - 0.5)**2 - 0.5)
        m.c2 = pe.Constraint(expr=m.y >= -(m.x + 2)**2 + 4)
        m.c3 = pe.Constraint(expr=m.y <= 2*m.x + 7)
        m.c4 = pe.Constraint(expr=m.y >= m.x)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        nlp_solver.config.log_level = logging.DEBUG
        opt = coramin.algorithms.MultiTree(mip_solver=mip_solver,
                                           nlp_solver=nlp_solver)
        opt.config.load_solution = False
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.infeasible)

    def test_all_vars_fixed_in_nlp(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(bounds=(-2, 1))
        m.y = pe.Var(domain=pe.Integers)
        m.z = pe.Var()
        m.obj = pe.Objective(expr=m.z - 0.2*m.y)
        m.c1 = pe.Constraint(expr=m.y == (m.x - 0.5)**2 - 0.5)
        m.c2 = pe.Constraint(expr=m.z == (m.x + 1)**2)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        nlp_solver.config.log_level = logging.DEBUG
        opt = coramin.algorithms.MultiTree(mip_solver=mip_solver,
                                           nlp_solver=nlp_solver)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self._check_relative_diff(-0.462486082, res.best_feasible_objective)
        self._check_relative_diff(-0.462486082, res.best_objective_bound)
        self._check_relative_diff(-1.37082869, m.x.value)
        self._check_relative_diff(3, m.y.value)
        self._check_relative_diff(0.137513918, m.z.value)

    def test_linear_problem(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        opt = coramin.algorithms.MultiTree(mip_solver=mip_solver, nlp_solver=nlp_solver)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        self.assertAlmostEqual(res.best_objective_bound, 1)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)

    def test_stale_fixed_vars(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.z = pe.Var(domain=pe.Binary)
        m.w = pe.Var()
        m.obj = pe.Objective(expr=m.y)
        m.c1 = pe.Constraint(expr=m.y >= m.x + 1)
        m.c2 = pe.Constraint(expr=m.y >= -m.x + 1)
        m.c3 = pe.Constraint(expr=m.w == 2)
        mip_solver = appsi.solvers.Gurobi()
        nlp_solver = appsi.solvers.Ipopt()
        opt = coramin.algorithms.MultiTree(mip_solver=mip_solver, nlp_solver=nlp_solver)
        res = opt.solve(m)
        self.assertEqual(res.termination_condition, appsi.base.TerminationCondition.optimal)
        self.assertAlmostEqual(res.best_feasible_objective, 1)
        self.assertAlmostEqual(res.best_objective_bound, 1)
        self.assertAlmostEqual(m.x.value, 0)
        self.assertAlmostEqual(m.y.value, 1)
        self.assertAlmostEqual(m.w.value, 2)
        self.assertIsNone(m.z.value)
