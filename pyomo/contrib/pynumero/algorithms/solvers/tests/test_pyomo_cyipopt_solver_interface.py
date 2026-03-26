from unittest import mock

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.pynumero.algorithms.solvers import cyipopt_solver


class _FakeNLP:
    def __init__(self, model):
        self.model = model

    def g_lb(self):
        return []

    def x_lb(self):
        return [0.0]

    def scaling_factors(self):
        return None, None, None

    def x_init(self):
        return [1.0]

    def set_primals(self, x):
        self.model.x.set_value(x[0])

    def set_duals(self, y):
        pass

    def load_state_into_pyomo(self, bound_multipliers=None):
        pass


class _FakeCyIpoptProblem:
    instances = []

    def __init__(self, nlp, intermediate_callback=None, halt_on_evaluation_error=None):
        self._nlp = nlp
        self.options = {}
        type(self).instances.append(self)

    def g_lb(self):
        return self._nlp.g_lb()

    def x_lb(self):
        return self._nlp.x_lb()

    def scaling_factors(self):
        return self._nlp.scaling_factors()

    def add_option(self, key, val):
        self.options[key] = val

    def solve(self, xstart):
        return [2.0], {
            'status': 0,
            'status_msg': (
                b"Algorithm terminated successfully at a locally "
                b"optimal point, satisfying the convergence tolerances "
                b"(can be specified by options)."
            ),
            'obj_val': 2.0,
            'mult_g': [],
            'mult_x_L': [0.0],
            'mult_x_U': [0.0],
        }

    def close(self):
        pass


class TestPyomoCyIpoptSolverInterface(unittest.TestCase):
    def setUp(self):
        _FakeCyIpoptProblem.instances.clear()
        self._patchers = [
            mock.patch.object(
                cyipopt_solver.pyomo_nlp, 'PyomoNLP', side_effect=_FakeNLP
            ),
            mock.patch.object(
                cyipopt_solver.cyipopt_interface,
                'CyIpoptNLP',
                side_effect=_FakeCyIpoptProblem,
            ),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self._patchers):
            patcher.stop()

    def _make_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var(initialize=1.0)
        m.obj = pyo.Objective(expr=m.x)
        return m

    def test_default_options_are_applied(self):
        solver = pyo.SolverFactory('cyipopt')
        solver.config.options['tol'] = 1e-6

        solver.solve(self._make_model())

        self.assertEqual(_FakeCyIpoptProblem.instances[-1].options, {'tol': 1e-6})

    def test_solve_solver_options_override_defaults_without_mutation(self):
        solver = pyo.SolverFactory('cyipopt')
        solver.config.options['tol'] = 1e-6
        solver.config.options['max_iter'] = 10

        solver.solve(
            self._make_model(),
            solver_options={'max_iter': 1, 'print_level': 0},
        )

        self.assertEqual(
            _FakeCyIpoptProblem.instances[-1].options,
            {'tol': 1e-6, 'max_iter': 1, 'print_level': 0},
        )
        self.assertEqual(
            dict(solver.config.options.items()), {'tol': 1e-6, 'max_iter': 10}
        )

        solver.solve(self._make_model())
        self.assertEqual(
            _FakeCyIpoptProblem.instances[-1].options,
            {'tol': 1e-6, 'max_iter': 10},
        )

    def test_legacy_options_keyword_is_supported(self):
        solver = pyo.SolverFactory('cyipopt')
        solver.config.options['tol'] = 1e-6

        solver.solve(self._make_model(), options={'max_iter': 3})

        self.assertEqual(
            _FakeCyIpoptProblem.instances[-1].options,
            {'tol': 1e-6, 'max_iter': 3},
        )
        self.assertEqual(dict(solver.config.options.items()), {'tol': 1e-6})

    def test_options_and_solver_options_cannot_both_be_passed(self):
        solver = pyo.SolverFactory('cyipopt')

        with self.assertRaisesRegex(ValueError, "Both 'options' and 'solver_options'"):
            solver.solve(
                self._make_model(),
                options={'max_iter': 3},
                solver_options={'max_iter': 1},
            )
