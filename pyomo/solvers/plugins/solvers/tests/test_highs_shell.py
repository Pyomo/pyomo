# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import os
import tempfile
import unittest

import pyomo.environ as pyo
from pyomo.opt.base import ProblemFormat
from pyomo.opt.results import SolverStatus, SolutionStatus, TerminationCondition
from pyomo.solvers.plugins.solvers.HIGHS import HIGHSSHELL


def _make_lp_model():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 1))
    m.y = pyo.Var(bounds=(0, 1))
    m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.maximize)
    m.c1 = pyo.Constraint(expr=m.x + m.y <= 1.5)
    return m


def _highs_available():
    s = HIGHSSHELL()
    return s.available(exception_flag=False)


_requires_highs = unittest.skipUnless(
    _highs_available(), "HiGHS executable not found on PATH"
)


class TestHighsSolverFactoryDispatch(unittest.TestCase):
    def test_default_returns_shell(self):
        s = pyo.SolverFactory('highs')
        self.assertIsInstance(s, HIGHSSHELL)

    def test_lp_mode_returns_shell(self):
        s = pyo.SolverFactory('highs', solver_io='lp')
        self.assertIsInstance(s, HIGHSSHELL)
        self.assertEqual(s._problem_format, ProblemFormat.cpxlp)

    def test_mps_mode_returns_shell(self):
        s = pyo.SolverFactory('highs', solver_io='mps')
        self.assertIsInstance(s, HIGHSSHELL)
        self.assertEqual(s._problem_format, ProblemFormat.mps)

    def test_direct_mode_not_shell(self):
        s = pyo.SolverFactory('highs', solver_io='direct')
        self.assertNotIsInstance(s, HIGHSSHELL)

    def test_highs_shell_name_registered(self):
        s = pyo.SolverFactory('_highs_shell')
        self.assertIsInstance(s, HIGHSSHELL)

    def test_unknown_mode_returns_unknown_solver(self):
        from pyomo.opt.base.solvers import UnknownSolver
        with self.assertLogs('pyomo.solvers', level='ERROR'):
            s = pyo.SolverFactory('highs', solver_io='nonexistent_mode')
        self.assertIsInstance(s, UnknownSolver)


class TestHighsShellCommandLine(unittest.TestCase):
    def setUp(self):
        self.solver = HIGHSSHELL()
        self._log_file = tempfile.NamedTemporaryFile(suffix='.highs.log', delete=False)
        self._log_file.close()
        self.solver._log_file = self._log_file.name

    def tearDown(self):
        os.unlink(self._log_file.name)

    def _make_cmd(self, problem_file, **opts):
        self.solver.options.update(opts)
        return self.solver.create_command_line(
            executable='highs', problem_files=[problem_file]
        )

    def test_basic_command_structure(self):
        b = self._make_cmd('model.lp')
        self.assertEqual(b.cmd[0], 'highs')
        self.assertEqual(b.cmd[1], 'model.lp')
        self.assertIn('--solution_file', b.cmd)

    def test_solution_file_derived_from_problem_file(self):
        b = self._make_cmd('/tmp/problem.lp')
        idx = b.cmd.index('--solution_file')
        self.assertEqual(b.cmd[idx + 1], '/tmp/problem.sol')

    def test_time_limit_added(self):
        self.solver._timelimit = 60
        b = self._make_cmd('model.lp')
        self.assertIn('--time_limit', b.cmd)
        self.assertEqual(b.cmd[b.cmd.index('--time_limit') + 1], '60')

    def test_extra_options_forwarded(self):
        b = self._make_cmd('model.lp', presolve='off', simplex_scale_strategy=2)
        self.assertIn('--presolve', b.cmd)
        self.assertIn('--simplex_scale_strategy', b.cmd)

    def test_solution_file_attribute_set(self):
        self._make_cmd('model.lp')
        self.assertTrue(self.solver._soln_file.endswith('.sol'))


class TestHighsShellLogfileParsing(unittest.TestCase):
    def setUp(self):
        self.solver = HIGHSSHELL()
        self._log_file = tempfile.NamedTemporaryFile(
            suffix='.highs.log', delete=False, mode='w'
        )
        self._log_file.close()
        self.solver._log_file = self._log_file.name

    def tearDown(self):
        os.unlink(self._log_file.name)

    def _write_log(self, content):
        with open(self._log_file.name, 'w') as f:
            f.write(content)

    def test_optimal(self):
        self._write_log(
            "Model status        : Optimal\n"
            "Objective value     :  1.5000000000e+00\n"
        )
        results = self.solver.process_logfile()
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertEqual(len(results.solution), 1)
        self.assertAlmostEqual(
            results.solution[0].objective['__default_objective__']['Value'], 1.5
        )

    def test_infeasible(self):
        self._write_log("Model status        : Infeasible\n")
        results = self.solver.process_logfile()
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(len(results.solution), 0)

    def test_unbounded(self):
        self._write_log("Model status        : Unbounded\n")
        results = self.solver.process_logfile()
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.unbounded
        )

    def test_infeasible_or_unbounded(self):
        self._write_log("Model status        : Infeasible or unbounded\n")
        results = self.solver.process_logfile()
        self.assertEqual(
            results.solver.termination_condition,
            TerminationCondition.infeasibleOrUnbounded,
        )

    def test_time_limit(self):
        self._write_log("Model status        : Time limit reached\n")
        results = self.solver.process_logfile()
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxTimeLimit
        )

    def test_iteration_limit(self):
        self._write_log("Model status        : Iteration limit reached\n")
        results = self.solver.process_logfile()
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxIterations
        )

    def test_empty_log_does_not_crash(self):
        self._write_log("")
        self.assertIsNotNone(self.solver.process_logfile())


class TestHighsShellSolnFileParsing(unittest.TestCase):
    _SAMPLE_SOL = (
        "Columns\n"
        "    Index Status        Lower        Upper       Primal         Dual  Name\n"
        "        0     BS            0            1          0.5           -0  x\n"
        "        1     UB            0            1            1           -0  y\n"
        "Rows\n"
        "    Index Status        Lower        Upper       Primal         Dual  Name\n"
        "        0     UB         -inf          1.5          1.5            1  c1\n"
        "\n"
        "Model status: Optimal\n"
    )

    def setUp(self):
        self.solver = HIGHSSHELL()
        self._sol_file = tempfile.NamedTemporaryFile(
            suffix='.sol', delete=False, mode='w'
        )
        self._sol_file.write(self._SAMPLE_SOL)
        self._sol_file.close()
        self.solver._soln_file = self._sol_file.name

    def tearDown(self):
        os.unlink(self._sol_file.name)

    def _results_with_soln(self):
        from pyomo.opt.results import SolverResults, Solution
        r = SolverResults()
        r.solution.insert(Solution())
        return r

    def test_variable_values_extracted(self):
        results = self._results_with_soln()
        self.solver._suffixes = []
        self.solver.process_soln_file(results)
        soln = results.solution[0]
        self.assertAlmostEqual(soln.variable['x']['Value'], 0.5)
        self.assertAlmostEqual(soln.variable['y']['Value'], 1.0)

    def test_duals_extracted_when_requested(self):
        results = self._results_with_soln()
        self.solver._suffixes = ['dual']
        self.solver.process_soln_file(results)
        self.assertAlmostEqual(results.solution[0].constraint['c1']['Dual'], 1.0)

    def test_reduced_costs_extracted_when_requested(self):
        results = self._results_with_soln()
        self.solver._suffixes = ['rc']
        self.solver.process_soln_file(results)
        self.assertIn('Rc', results.solution[0].variable['x'])

    def test_missing_soln_file_does_not_crash(self):
        self.solver._soln_file = '/nonexistent/path.sol'
        results = self._results_with_soln()
        self.solver._suffixes = []
        self.solver.process_soln_file(results)

    def test_empty_solution_list_skipped(self):
        from pyomo.opt.results import SolverResults
        results = SolverResults()
        self.solver._suffixes = []
        self.solver.process_soln_file(results)


@_requires_highs
class TestHighsShellSolveRoundTrip(unittest.TestCase):
    def _solve(self, model, **kwargs):
        return pyo.SolverFactory('highs').solve(model, **kwargs)

    def test_optimal_lp(self):
        m = _make_lp_model()
        res = self._solve(m)
        self.assertEqual(res.solver.status, SolverStatus.ok)
        self.assertEqual(
            res.solver.termination_condition, TerminationCondition.optimal
        )
        self.assertAlmostEqual(pyo.value(m.x) + pyo.value(m.y), 1.5, places=4)

    def test_objective_value(self):
        m = _make_lp_model()
        res = self._solve(m)
        obj_val = res.solution[0].objective['__default_objective__']['Value']
        self.assertAlmostEqual(obj_val, 1.5, places=4)

    def test_mps_mode(self):
        m = _make_lp_model()
        opt = pyo.SolverFactory('highs', solver_io='mps')
        res = opt.solve(m)
        self.assertEqual(
            res.solver.termination_condition, TerminationCondition.optimal
        )

    def test_infeasible_lp(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(bounds=(0, 1))
        m.obj = pyo.Objective(expr=m.x)
        m.c1 = pyo.Constraint(expr=m.x >= 2)
        res = self._solve(m)
        self.assertEqual(
            res.solver.termination_condition, TerminationCondition.infeasible
        )

    def test_timelimit_respected(self):
        m = _make_lp_model()
        opt = pyo.SolverFactory('highs')
        opt._timelimit = 0.001
        res = opt.solve(m)
        self.assertIn(
            res.solver.termination_condition,
            {TerminationCondition.optimal, TerminationCondition.maxTimeLimit},
        )


if __name__ == '__main__':
    unittest.main()
