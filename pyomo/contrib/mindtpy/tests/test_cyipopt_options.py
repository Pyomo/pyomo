from types import SimpleNamespace
from unittest import mock

import pyomo.common.unittest as unittest

from pyomo.contrib.mindtpy import util


class TestCyIpoptSolveOptions(unittest.TestCase):
    def test_update_solver_timelimit_uses_solve_time_solver_options(self):
        solver = SimpleNamespace(config=SimpleNamespace(options={}), options={})
        config = SimpleNamespace(time_limit=10)
        solve_args = {'solver_options': {'print_level': 0}}

        with mock.patch.object(util, 'get_main_elapsed_time', return_value=3.2):
            util.update_solver_timelimit(
                solver, 'cyipopt', timing=object(), config=config, solve_args=solve_args
            )

        self.assertEqual(
            solve_args['solver_options'],
            {'print_level': 0, 'max_cpu_time': 7.0},
        )
        self.assertEqual(solver.config.options, {})

    def test_constraint_tolerance_uses_existing_legacy_options_keyword(self):
        solver = SimpleNamespace(config=SimpleNamespace(options={}), options={})
        config = SimpleNamespace(zero_tolerance=1e-7)
        solve_args = {'options': {'max_iter': 5}}

        util.set_solver_constraint_violation_tolerance(
            solver, 'cyipopt', config, solve_args=solve_args
        )

        self.assertEqual(
            solve_args['options'],
            {'max_iter': 5, 'constr_viol_tol': 1e-7},
        )
        self.assertEqual(solver.config.options, {})

    def test_solver_option_merge_rejects_mixed_option_keywords(self):
        with self.assertRaisesRegex(ValueError, "Both 'options' and 'solver_options'"):
            util._update_solve_solver_options(
                {'options': {}, 'solver_options': {}}, {'max_cpu_time': 1.0}
            )
