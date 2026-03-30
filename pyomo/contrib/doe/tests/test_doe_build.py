# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import json
import os
import os.path
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)

from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest
from pyomo.contrib.doe.doe import ObjectiveLib, _DoEResultsJSONEncoder

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

if scipy_available:
    from pyomo.contrib.doe import DesignOfExperiments
    from pyomo.contrib.doe.examples.reactor_example import (
        ReactorExperiment as FullReactorExperiment,
    )
    from pyomo.contrib.doe.tests.experiment_class_example_flags import (
        RooneyBieglerMultiExperiment,
    )
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
        RooneyBieglerExperiment,
    )

from pyomo.contrib.doe.examples.rooney_biegler_doe_example import run_rooney_biegler_doe
import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)

data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}


def get_rooney_biegler_experiment():
    """Get a fresh RooneyBieglerExperiment instance for testing.

    Creates a new experiment instance to ensure test isolation.
    Each test gets its own instance to avoid state sharing.
    """
    data = pd.DataFrame(data=[[5, 15.6]], columns=['hour', 'y'])
    data_point = data.iloc[0]

    return RooneyBieglerExperiment(
        data=data_point,
        theta={'asymptote': 15, 'rate_constant': 0.5},
        measure_error=0.1,
    )


def get_FIM_FIMPrior_Q_L(doe_obj=None):
    """
    Helper function to retrieve results to compare.

    """
    model = doe_obj.model

    n_param = doe_obj.n_parameters
    n_y = doe_obj.n_experiment_outputs

    FIM_vals = [
        pyo.value(model.fim[i, j])
        for i in model.parameter_names
        for j in model.parameter_names
    ]
    FIM_prior_vals = [
        pyo.value(model.prior_FIM[i, j])
        for i in model.parameter_names
        for j in model.parameter_names
    ]
    if hasattr(model, "L"):
        L_vals = [
            pyo.value(model.L[i, j])
            for i in model.parameter_names
            for j in model.parameter_names
        ]
    else:
        L_vals = [[0] * n_param] * n_param
    Q_vals = [
        pyo.value(model.sensitivity_jacobian[i, j])
        for i in model.output_names
        for j in model.parameter_names
    ]
    sigma_inv = [
        1 / v for k, v in model.fd_scenario_blocks[0].measurement_error.items()
    ]
    param_vals = np.array(
        [[v for k, v in model.fd_scenario_blocks[0].unknown_parameters.items()]]
    )

    FIM_vals_np = np.array(FIM_vals).reshape((n_param, n_param))
    FIM_prior_vals_np = np.array(FIM_prior_vals).reshape((n_param, n_param))

    for i in range(n_param):
        for j in range(n_param):
            if j < i:
                FIM_vals_np[j, i] = FIM_vals_np[i, j]

    L_vals_np = np.array(L_vals).reshape((n_param, n_param))
    Q_vals_np = np.array(Q_vals).reshape((n_y, n_param))

    sigma_inv_np = np.zeros((n_y, n_y))

    for ind, v in enumerate(sigma_inv):
        sigma_inv_np[ind, ind] = v

    return FIM_vals_np, FIM_prior_vals_np, Q_vals_np, L_vals_np, sigma_inv_np


def get_standard_args(experiment, fd_method, obj_used):
    args = {}
    args['experiment'] = None if experiment is None else [experiment]
    args['fd_formula'] = fd_method
    args['step'] = 1e-3
    args['objective_option'] = obj_used
    args['scale_constant_value'] = 1
    args['scale_nominal_param_value'] = True
    args['prior_FIM'] = None
    args['jac_initial'] = None
    args['fim_initial'] = None
    args['L_diagonal_lower_bound'] = 1e-7
    # Make solver object with
    # good linear subroutines
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 3000
    args['solver'] = solver
    args['tee'] = False
    args['get_labeled_model_args'] = None
    args['_Cholesky_option'] = True
    args['_only_compute_fim_lower'] = True
    return args


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestDoeBuild(unittest.TestCase):
    def test_constructor_accepts_single_experiment_or_list(self):
        # The public constructor should normalize either form into the same
        # internal experiment_list representation.
        single = DesignOfExperiments(
            experiment=RooneyBieglerMultiExperiment(hour=2.0),
            objective_option="pseudo_trace",
        )
        as_list = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
        )
        self.assertEqual(len(single.experiment_list), 1)
        self.assertEqual(len(as_list.experiment_list), 1)

    def test_rooney_biegler_fd_central_check_fd_eqns(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the parameter values are correct
        for s in model.scenarios:
            param = model.parameter_scenarios[s]

            diff = (-1) ** s * doe_obj.step

            param_val = pyo.value(
                pyo.ComponentUID(param).find_component_on(model.fd_scenario_blocks[s])
            )

            param_val_from_step = model.fd_scenario_blocks[0].unknown_parameters[
                pyo.ComponentUID(param).find_component_on(model.fd_scenario_blocks[0])
            ] * (1 + diff)

            for k, v in model.fd_scenario_blocks[s].unknown_parameters.items():
                if pyo.ComponentUID(
                    k, context=model.fd_scenario_blocks[s]
                ) == pyo.ComponentUID(param):
                    continue

                other_param_val = pyo.value(k)
                self.assertAlmostEqual(other_param_val, v)

            self.assertAlmostEqual(param_val, param_val_from_step)

    def test_rooney_biegler_fd_backward_check_fd_eqns(self):
        fd_method = "backward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the parameter values are correct
        for s in model.scenarios:
            diff = -doe_obj.step * (s != 0)
            if s != 0:
                param = model.parameter_scenarios[s]

                param_val = pyo.value(
                    pyo.ComponentUID(param).find_component_on(
                        model.fd_scenario_blocks[s]
                    )
                )

                param_val_from_step = model.fd_scenario_blocks[0].unknown_parameters[
                    pyo.ComponentUID(param).find_component_on(
                        model.fd_scenario_blocks[0]
                    )
                ] * (1 + diff)
                self.assertAlmostEqual(param_val, param_val_from_step)

            for k, v in model.fd_scenario_blocks[s].unknown_parameters.items():
                if (s != 0) and pyo.ComponentUID(
                    k, context=model.fd_scenario_blocks[s]
                ) == pyo.ComponentUID(param):
                    continue

                other_param_val = pyo.value(k)
                self.assertAlmostEqual(other_param_val, v)

    def test_rooney_biegler_fd_forward_check_fd_eqns(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the parameter values are correct
        for s in model.scenarios:
            diff = doe_obj.step * (s != 0)
            if s != 0:
                param = model.parameter_scenarios[s]

                param_val = pyo.value(
                    pyo.ComponentUID(param).find_component_on(
                        model.fd_scenario_blocks[s]
                    )
                )

                param_val_from_step = model.fd_scenario_blocks[0].unknown_parameters[
                    pyo.ComponentUID(param).find_component_on(
                        model.fd_scenario_blocks[0]
                    )
                ] * (1 + diff)
                self.assertAlmostEqual(param_val, param_val_from_step)

            for k, v in model.fd_scenario_blocks[s].unknown_parameters.items():
                if (s != 0) and pyo.ComponentUID(
                    k, context=model.fd_scenario_blocks[s]
                ) == pyo.ComponentUID(param):
                    continue

                other_param_val = pyo.value(k)
                self.assertAlmostEqual(other_param_val, v)

    def test_rooney_biegler_fd_central_design_fixing(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the design fixing constraints are generated
        design_vars = [
            k for k, v in model.fd_scenario_blocks[0].experiment_inputs.items()
        ]

        con_name_base = "global_design_eq_con_"

        # Ensure that
        for ind, d in enumerate(design_vars):
            if ind == 0:
                continue

            con_name = con_name_base + str(ind)
            self.assertTrue(hasattr(model, con_name))
            # Ensure that each set of constraints has all blocks pairs with scenario 0
            # i.e., (0, 1), (0, 2), ..., (0, N) --> N - 1 constraints
            self.assertEqual(len(getattr(model, con_name)), (len(model.scenarios) - 1))
            # Should not have any constraints sets beyond the
            # length of design_vars - 1 (started with index 0)
        self.assertFalse(hasattr(model, con_name_base + str(len(design_vars))))

    def test_rooney_biegler_fd_backward_design_fixing(self):
        fd_method = "backward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the design fixing constraints are generated
        design_vars = [
            k for k, v in model.fd_scenario_blocks[0].experiment_inputs.items()
        ]

        con_name_base = "global_design_eq_con_"

        # Ensure that
        for ind, d in enumerate(design_vars):
            if ind == 0:
                continue

            con_name = con_name_base + str(ind)
            self.assertTrue(hasattr(model, con_name))
            # Ensure that each set of constraints has all blocks pairs with scenario 0
            # i.e., (0, 1), (0, 2), ..., (0, N) --> N - 1 constraints
            self.assertEqual(len(getattr(model, con_name)), (len(model.scenarios) - 1))
            # Should not have any constraints sets beyond the
            # length of design_vars - 1 (started with index 0)
        self.assertFalse(hasattr(model, con_name_base + str(len(design_vars))))

    def test_rooney_biegler_fd_forward_design_fixing(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the design fixing constraints are generated
        design_vars = [
            k for k, v in model.fd_scenario_blocks[0].experiment_inputs.items()
        ]

        con_name_base = "global_design_eq_con_"

        # Ensure that
        for ind, d in enumerate(design_vars):
            if ind == 0:
                continue

            con_name = con_name_base + str(ind)
            self.assertTrue(hasattr(model, con_name))
            # Ensure that each set of constraints has all blocks pairs with scenario 0
            # i.e., (0, 1), (0, 2), ..., (0, N) --> N - 1 constraints
            self.assertEqual(len(getattr(model, con_name)), (len(model.scenarios) - 1))
            # Should not have any constraints sets beyond the
            # length of design_vars - 1 (started with index 0)
        self.assertFalse(hasattr(model, con_name_base + str(len(design_vars))))

    def test_rooney_biegler_check_user_initialization(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment()

        FIM_prior = np.ones((2, 2))
        FIM_initial = np.eye(2) + FIM_prior
        JAC_initial = np.ones((1, 2)) * 2

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args['prior_FIM'] = FIM_prior
        DoE_args['fim_initial'] = FIM_initial
        DoE_args['jac_initial'] = JAC_initial

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        # Grab the matrix values on the model
        FIM, FIM_prior_model, Q, L, sigma = get_FIM_FIMPrior_Q_L(doe_obj)

        # Make sure they match the inputs we gave
        assert np.array_equal(FIM, FIM_initial)
        assert np.array_equal(FIM_prior, FIM_prior_model)
        assert np.array_equal(JAC_initial, Q)

    def test_update_FIM(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        FIM_update = np.ones((2, 2)) * 10

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        doe_obj.update_FIM_prior(FIM=FIM_update)

        # Grab values to ensure we set the correct piece
        FIM, FIM_prior_model, Q, L, sigma = get_FIM_FIMPrior_Q_L(doe_obj)

        # Make sure they match the inputs we gave
        assert np.array_equal(FIM_update, FIM_prior_model)

    def test_get_experiment_inputs_without_blocks(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

        stuff = doe_obj.get_experiment_input_values(model=doe_obj.compute_FIM_model)

        count = 0
        for k, v in doe_obj.compute_FIM_model.experiment_inputs.items():
            self.assertEqual(pyo.value(k), stuff[count])
            count += 1

    def test_get_experiment_outputs_without_blocks(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

        stuff = doe_obj.get_experiment_output_values(model=doe_obj.compute_FIM_model)

        count = 0
        for k, v in doe_obj.compute_FIM_model.experiment_outputs.items():
            self.assertEqual(pyo.value(k), stuff[count])
            count += 1

    def test_get_measurement_error_without_blocks(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

        stuff = doe_obj.get_measurement_error_values(model=doe_obj.compute_FIM_model)

        count = 0
        for k, v in doe_obj.compute_FIM_model.measurement_error.items():
            self.assertEqual(pyo.value(k), stuff[count])
            count += 1

    def test_get_unknown_parameters_without_blocks(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

        # Make sure the values can be retrieved
        stuff = doe_obj.get_unknown_parameter_values(model=doe_obj.compute_FIM_model)

        count = 0
        for k, v in doe_obj.compute_FIM_model.unknown_parameters.items():
            self.assertEqual(pyo.value(k), stuff[count])
            count += 1

    def test_generate_blocks_without_model(self):
        fd_method = "forward"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment()

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj._generate_fd_scenario_blocks()

        for i in doe_obj.model.parameter_scenarios:
            self.assertTrue(
                doe_obj.model.find_component("fd_scenario_blocks[" + str(i) + "]")
            )


class TestReactorExample(unittest.TestCase):
    def test_reactor_update_suffix_items(self):
        """Test the reactor example with updating suffix items."""
        from pyomo.contrib.doe.examples.update_suffix_doe_example import main

        # Run the reactor update suffix items example
        suffix_obj, _, new_vals = main()

        # Check that the suffix object has been updated correctly
        for i, v in enumerate(suffix_obj.values()):
            self.assertAlmostEqual(v, new_vals[i], places=6)


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestDoEObjectiveOptions(unittest.TestCase):
    def test_maximize_objective_set_contents(self):
        # The objective-sense partition drives maximize/minimize scoring logic
        # in initialization and solve helpers, so keep a direct regression on
        # the public enum membership of the maximize set.
        maximize = DesignOfExperiments._MAXIMIZE_OBJECTIVES
        self.assertIn(ObjectiveLib.determinant, maximize)
        self.assertIn(ObjectiveLib.pseudo_trace, maximize)
        self.assertIn(ObjectiveLib.minimum_eigenvalue, maximize)
        self.assertNotIn(ObjectiveLib.trace, maximize)
        self.assertNotIn(ObjectiveLib.condition_number, maximize)
        self.assertNotIn(ObjectiveLib.zero, maximize)

    def test_trace_constraints(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = run_rooney_biegler_doe(optimization_objective="trace")[
            "experiment"
        ]

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()
        doe_obj.create_objective_function()

        model = doe_obj.model

        # Basic objects exist
        self.assertTrue(hasattr(model, "objective"))
        self.assertTrue(hasattr(model, "cov_trace"))
        self.assertTrue(hasattr(model, "fim_inv"))
        self.assertTrue(hasattr(model, "L"))
        self.assertTrue(hasattr(model, "L_inv"))

        # Constraints live under obj_cons block
        self.assertTrue(hasattr(model, "obj_cons"))

        # Cholesky-related constraints
        self.assertTrue(hasattr(model.obj_cons, "cholesky_cons"))
        self.assertTrue(hasattr(model.obj_cons, "cholesky_inv_cons"))
        self.assertTrue(hasattr(model.obj_cons, "cholesky_LLinv_cons"))
        self.assertTrue(hasattr(model.obj_cons, "cov_trace_rule"))

        self.assertIsInstance(model.obj_cons.cholesky_cons, pyo.Constraint)
        self.assertIsInstance(model.obj_cons.cholesky_inv_cons, pyo.Constraint)
        self.assertIsInstance(model.obj_cons.cholesky_LLinv_cons, pyo.Constraint)
        self.assertIsInstance(model.obj_cons.cov_trace_rule, pyo.Constraint)

        # Indexing logic: lower triangle only
        params = list(model.parameter_names)

        for i, c in enumerate(params):
            for j, d in enumerate(params):
                # inverse constraint only exists for lower triangle (i >= j)
                if i >= j:
                    self.assertIn(
                        (c, d),
                        model.obj_cons.cholesky_inv_cons,
                        msg=f"Missing cholesky_inv_cons[{c},{d}]",
                    )
                else:
                    self.assertNotIn(
                        (c, d),
                        model.obj_cons.cholesky_inv_cons,
                        msg=f"Unexpected cholesky_inv_cons[{c},{d}]",
                    )

                # identity constraint only defined for lower triangle (i >= j)
                if i >= j:
                    self.assertIn(
                        (c, d),
                        model.obj_cons.cholesky_LLinv_cons,
                        msg=f"Missing cholesky_LLinv_cons[{c},{d}]",
                    )
                else:
                    self.assertNotIn(
                        (c, d),
                        model.obj_cons.cholesky_LLinv_cons,
                        msg=f"Unexpected cholesky_LLinv_cons[{c},{d}]",
                    )

        # Objective definition sanity
        self.assertIsInstance(model.objective, pyo.Objective)
        self.assertEqual(model.objective.sense, pyo.minimize)
        self.assertIs(model.objective.expr, model.cov_trace)

    def test_trace_initialization_consistency(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = run_rooney_biegler_doe(optimization_objective="trace")[
            "experiment"
        ]

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()
        doe_obj.create_objective_function()

        model = doe_obj.model
        params = list(model.parameter_names)

        # Check cov_trace initialization
        cov_trace_expected = 2.0
        self.assertAlmostEqual(pyo.value(model.cov_trace), cov_trace_expected, places=4)

        # Check L * L_inv ≈ I (lower triangle)
        for i, c in enumerate(params):
            for j, d in enumerate(params):
                if i < j:
                    continue  # upper triangle skipped by design

                val = pyo.value(
                    sum(
                        model.L[c, params[k]] * model.L_inv[params[k], d]
                        for k in range(len(params))
                    )
                )

                expected = 1.0 if i == j else 0.0
                self.assertAlmostEqual(val, expected, places=4)


class TestOptimizeExperimentsBuildStructure(unittest.TestCase):
    """Coverage for optimize_experiments() build, output, and diagnostics behavior."""

    def _make_solver(self):
        # Make solver object with options for DoE runs.
        solver = SolverFactory("ipopt")
        solver.options["linear_solver"] = "ma57"
        solver.options["halt_on_ampl_error"] = "yes"
        solver.options["max_iter"] = 3000
        return solver

    def _mock_solver_results(self, message):
        return SimpleNamespace(
            solver=SimpleNamespace(
                status="ok",
                termination_condition="optimal",
                message=message,
            )
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_init_solver_used_for_initialization_only(self):
        # Tests that all pre-final solves use init_solver while the final
        # optimization solve still uses the primary solver.
        main_solver = self._make_solver()
        init_solver = self._make_solver()
        # Use distinct option values so each solver path can be identified.
        main_solver.options["max_iter"] = 321
        init_solver.options["max_iter"] = 123
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=main_solver,
        )

        # Track both how many times each solver is called and the chronological
        # order of those calls. The optimize_experiments() contract is that all
        # setup solves run first on init_solver and the final NLP solve runs last
        # on the primary solver.
        main_calls = 0
        init_calls = 0
        call_order = []
        # Record an option value on each solve so the test can verify that the
        # call really went through the expected solver object, not just the
        # expected phase label.
        option_markers = []
        original_main_solve = main_solver.solve
        original_init_solve = init_solver.solve

        def _main_solve(*args, **kwargs):
            nonlocal main_calls
            main_calls += 1
            call_order.append("main")
            option_markers.append(main_solver.options.get("max_iter"))
            return original_main_solve(*args, **kwargs)

        def _init_solve(*args, **kwargs):
            nonlocal init_calls
            init_calls += 1
            call_order.append("init")
            option_markers.append(init_solver.options.get("max_iter"))
            return original_init_solve(*args, **kwargs)

        # Patch both solver objects in place so the real solves still run while
        # we collect lightweight diagnostics about solver routing.
        with (
            patch.object(main_solver, "solve", side_effect=_main_solve),
            patch.object(init_solver, "solve", side_effect=_init_solve),
        ):
            doe_obj.optimize_experiments(n_exp=2, init_solver=init_solver)

        # The exact number of initialization solves is implementation-dependent,
        # but they must all occur before the one final main-solver call.
        self.assertGreaterEqual(init_calls, 1)  # At least one initialization solve
        self.assertEqual(main_calls, 1)  # Exactly one main optimization solve
        self.assertEqual(call_order[-1], "main")
        self.assertTrue(all(tag == "init" for tag in call_order[:-1]))
        # Distinct option markers provide a second check that solver routing
        # matches the expected init-versus-final phase split.
        self.assertTrue(all(marker == 123 for marker in option_markers[:-1]))
        self.assertEqual(option_markers[-1], 321)
        # Result payloads should report the same phase-specific solver names that
        # were observed through the patched solve() calls above.
        self.assertEqual(
            doe_obj.results["settings"]["initialization"]["solver_name"],
            getattr(init_solver, "name", str(init_solver)),
        )
        self.assertEqual(
            doe_obj.results["run_info"]["solver"]["name"],
            getattr(main_solver, "name", str(main_solver)),
        )

    def test_get_experiment_input_vars_direct_and_fd_fallback(self):
        # Test the helper used by optimize_experiments() finds input vars for both
        # direct models and finite-difference scenario-block models.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
        )

        model_direct = pyo.ConcreteModel()
        model_direct.x = pyo.Var(initialize=2.0, bounds=(1.0, 10.0))
        model_direct.experiment_inputs = pyo.Suffix(direction=pyo.Suffix.LOCAL)
        model_direct.experiment_inputs[model_direct.x] = 2.0
        vars_direct = doe_obj._get_experiment_input_vars(model_direct)
        self.assertEqual([v.name for v in vars_direct], [model_direct.x.name])

        model_fd = pyo.ConcreteModel()
        model_fd.fd_scenario_blocks = pyo.Block([0])
        model_fd.fd_scenario_blocks[0].x = pyo.Var(initialize=3.0, bounds=(1.0, 10.0))
        model_fd.fd_scenario_blocks[0].experiment_inputs = pyo.Suffix(
            direction=pyo.Suffix.LOCAL
        )
        model_fd.fd_scenario_blocks[0].experiment_inputs[
            model_fd.fd_scenario_blocks[0].x
        ] = 3.0
        vars_fd = doe_obj._get_experiment_input_vars(model_fd)
        self.assertEqual(
            [v.name for v in vars_fd], [model_fd.fd_scenario_blocks[0].x.name]
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_multi_experiment_structure_and_results(self):
        # Test that the multi-experiment optimize_experiments() run builds the expected
        # scenario/experiment structure and structured results.
        solver = self._make_solver()

        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiExperiment(hour=2.0),
                RooneyBieglerMultiExperiment(hour=4.0),
            ],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=solver,
        )
        doe_obj.optimize_experiments()

        scenario_block = doe_obj.model.param_scenario_blocks[0]
        self.assertTrue(hasattr(scenario_block, "symmetry_breaking_s0_exp1"))
        self.assertEqual(len(list(scenario_block.exp_blocks.keys())), 2)

        param_scenario = doe_obj.results["param_scenarios"][0]
        self.assertEqual(doe_obj.results["Initialization Method"], "none")
        self.assertEqual(doe_obj.results["Number of Experiments per Scenario"], 2)
        self.assertEqual(len(param_scenario["experiments"]), 2)
        self.assertEqual(len(doe_obj.results["Experiment Design Names"]), 1)
        self.assertEqual(len(doe_obj.results["Unknown Parameter Names"]), 2)

        # Results should expose a single structured parameter-scenario payload.
        self.assertIn("run_info", doe_obj.results)
        self.assertIn("settings", doe_obj.results)
        self.assertIn("timing", doe_obj.results)
        self.assertIn("names", doe_obj.results)
        self.assertIn("param_scenarios", doe_obj.results)
        self.assertEqual(
            doe_obj.results["run_info"]["solver"]["status"],
            doe_obj.results["Solver Status"],
        )
        self.assertEqual(
            doe_obj.results["settings"]["modeling"]["n_experiments_per_scenario"], 2
        )
        self.assertFalse(doe_obj.results["settings"]["modeling"]["template_mode"])
        self.assertEqual(len(doe_obj.results["param_scenarios"]), 1)
        self.assertEqual(doe_obj.results["param_scenarios"][0]["id"], 0)
        self.assertEqual(len(doe_obj.results["param_scenarios"][0]["experiments"]), 2)
        self.assertEqual(
            doe_obj.results["param_scenarios"][0]["experiments"][0]["id"], 0
        )

        # hour of exp[0] should be <= hour of exp[1] due to symmetry breaking
        h0 = param_scenario["experiments"][0]["design"][0]
        h1 = param_scenario["experiments"][1]["design"][0]
        self.assertLessEqual(h0, h1)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_writes_results_file(self):
        # Tests that optimize_experiments() writes JSON results when given either
        # a string path or a pathlib.Path for results_file.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        fd, results_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(
            lambda: os.path.exists(results_path) and os.remove(results_path)
        )

        doe_obj.optimize_experiments(n_exp=1, results_file=results_path)

        with open(results_path) as f:
            payload = json.load(f)
        self.assertEqual(payload["Initialization Method"], "none")
        self.assertTrue(payload["settings"]["modeling"]["template_mode"])
        self.assertIn("param_scenarios", payload)
        self.assertIn("run_info", payload)
        self.assertIn("settings", payload)
        self.assertIn("timing", payload)
        self.assertIn("names", payload)

        path_payload = Path(results_path)
        doe_obj.optimize_experiments(n_exp=1, results_file=path_payload)
        with open(results_path) as f:
            payload_path = json.load(f)
        self.assertEqual(payload_path["Initialization Method"], "none")
        self.assertTrue(payload_path["settings"]["modeling"]["template_mode"])

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_single_experiment_defaults_to_template_mode(self):
        # Tests that optimize_experiments() uses template mode by default when
        # n_exp=1.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        doe_obj.optimize_experiments()
        self.assertEqual(doe_obj.results["Number of Experiments per Scenario"], 1)
        self.assertTrue(doe_obj.results["settings"]["modeling"]["template_mode"])

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_zero_objective_works_without_obj_cons(self):
        # The zero objective intentionally builds no scenario.obj_cons block, so
        # optimize_experiments() must skip the deactivate/reactivate cycle and
        # still produce the usual results payload from the square solve.
        init_solver = self._make_solver()
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="zero",
            step=1e-2,
            solver=self._make_solver(),
        )
        final_calls = {"n": 0}

        def _mock_final_solve(*args, **kwargs):
            final_calls["n"] += 1
            return self._mock_solver_results("mock-zero-final")

        with patch.object(
            doe_obj.solver, "solve", side_effect=_mock_final_solve
        ):
            doe_obj.optimize_experiments(n_exp=2, init_solver=init_solver)

        scenario = doe_obj.model.param_scenario_blocks[0]
        self.assertFalse(hasattr(scenario, "obj_cons"))
        self.assertEqual(final_calls["n"], 1)
        self.assertEqual(doe_obj.results["Solver Status"], "ok")
        self.assertEqual(doe_obj.results["Termination Message"], "mock-zero-final")
        self.assertEqual(len(doe_obj.results["param_scenarios"][0]["experiments"]), 2)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_trace_roundoff_flag_builds_extra_constraints(self):
        # The multi-experiment trace path optionally adds extra Cholesky/FIM
        # diagonal constraints to reduce roundoff drift. Keep the square-solve
        # build real, but mock the final NLP solve because this test is only
        # checking that the additional constraints were created.
        init_solver = self._make_solver()
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="trace",
            step=1e-2,
            solver=self._make_solver(),
            improve_cholesky_roundoff_error=True,
        )
        final_calls = {"n": 0}

        def _mock_final_solve(*args, **kwargs):
            final_calls["n"] += 1
            return self._mock_solver_results("mock-trace-roundoff-final")

        with patch.object(doe_obj.solver, "solve", side_effect=_mock_final_solve):
            doe_obj.optimize_experiments(n_exp=2, init_solver=init_solver)

        scenario = doe_obj.model.param_scenario_blocks[0]
        parameter_names = list(scenario.exp_blocks[0].parameter_names)

        self.assertEqual(final_calls["n"], 1)
        self.assertTrue(hasattr(scenario.obj_cons, "cholesky_fim_diag_cons"))
        self.assertTrue(hasattr(scenario.obj_cons, "cholesky_fim_inv_diag_cons"))
        self.assertEqual(
            len(scenario.obj_cons.cholesky_fim_diag_cons), len(parameter_names) ** 2
        )
        self.assertEqual(
            len(scenario.obj_cons.cholesky_fim_inv_diag_cons),
            len(parameter_names) ** 2,
        )
        self.assertEqual(
            doe_obj.results["Termination Message"], "mock-trace-roundoff-final"
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_timing_includes_lhs_phase_separately(self):
        # Tests that LHS initialization timing is tracked separately and
        # contributes additively to total runtime accounting.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        doe_obj.optimize_experiments(
            n_exp=2, init_method="lhs", init_n_samples=2, init_seed=11
        )

        timing = doe_obj.results["timing"]
        self.assertIn("lhs_initialization_s", timing)
        self.assertGreaterEqual(timing["lhs_initialization_s"], 0.0)
        self.assertAlmostEqual(
            timing["total_s"],
            timing["build_s"]
            + timing["lhs_initialization_s"]
            + timing["initialization_s"]
            + timing["solve_s"],
            places=8,
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_symmetry_log_once_per_scenario(self):
        # Tests that symmetry-breaking constraint logging is emitted once per
        # scenario (not once per generated constraint).
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        with self.assertLogs("pyomo.contrib.doe.doe", level="INFO") as log_cm:
            doe_obj.optimize_experiments(n_exp=3)

        matching = [
            m
            for m in log_cm.output
            if "Added 2 symmetry breaking constraints for scenario 0" in m
        ]
        self.assertEqual(len(matching), 1)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_lhs_diagnostics_populated(self):
        # Tests that threaded LHS initialization records diagnostics needed for
        # debugging and performance visibility.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        doe_obj.optimize_experiments(
            n_exp=2,
            init_method="lhs",
            init_n_samples=2,
            init_seed=11,
            init_parallel=True,
            init_combo_parallel=True,
            init_n_workers=2,
            init_combo_chunk_size=2,
            init_combo_parallel_threshold=1,
            init_max_wall_clock_time=60.0,
        )
        lhs_diag = doe_obj.results["diagnostics"]["lhs_initialization"]
        self.assertIsNotNone(lhs_diag)
        self.assertEqual(lhs_diag["candidate_fim_mode"], "thread")
        self.assertEqual(lhs_diag["combo_mode"], "thread")
        self.assertEqual(lhs_diag["n_workers"], 2)
        self.assertFalse(lhs_diag["timed_out"])
        self.assertGreater(lhs_diag["elapsed_total_s"], 0.0)
        self.assertIn("best_obj", lhs_diag)
        self.assertIsInstance(lhs_diag["best_obj"], float)
        self.assertTrue(np.isfinite(lhs_diag["best_obj"]))
        self.assertGreater(lhs_diag["best_obj"], 0.0)
        self.assertIn("best_obj_log10", lhs_diag)
        self.assertIsInstance(lhs_diag["best_obj_log10"], float)
        self.assertAlmostEqual(
            lhs_diag["best_obj_log10"], np.log10(lhs_diag["best_obj"]), places=12
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_termination_message_bytes(self):
        # Tests that solver termination messages returned as bytes are decoded
        # and persisted as plain strings in results.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        original_solve = doe_obj.solver.solve

        def _solve_with_bytes_message(*args, **kwargs):
            res = original_solve(*args, **kwargs)
            res.solver.message = b"byte-message"
            return res

        with patch.object(
            doe_obj.solver, "solve", side_effect=_solve_with_bytes_message
        ):
            doe_obj.optimize_experiments(n_exp=1)

        self.assertEqual(doe_obj.results["Termination Message"], "byte-message")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_optimize_experiments_termination_message_fallback_to_str(self):
        # Tests that non-string termination messages fall back to str(message) so
        # results always store a serializable user-facing value.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        original_solve = doe_obj.solver.solve

        class _CustomMessage:
            def __str__(self):
                return "custom-message"

        def _solve_with_custom_message(*args, **kwargs):
            res = original_solve(*args, **kwargs)
            res.solver.message = _CustomMessage()
            return res

        with patch.object(
            doe_obj.solver, "solve", side_effect=_solve_with_custom_message
        ):
            doe_obj.optimize_experiments(n_exp=1)

        self.assertEqual(doe_obj.results["Termination Message"], "custom-message")

class TestDoeResultsSerialization(unittest.TestCase):
    """Coverage for DoE results payload serialization helpers."""

    def test_doe_results_json_encoder_handles_numpy_and_enum(self):
        # Results payloads are written to JSON from optimize_experiments(), so
        # the encoder must normalize numpy scalars/arrays and ObjectiveLib enums.
        payload = {
            "scalar": np.int64(7),
            "array": np.array([1.0, 2.0]),
            "objective": ObjectiveLib.trace,
        }
        encoded = json.dumps(payload, cls=_DoEResultsJSONEncoder)
        decoded = json.loads(encoded)

        self.assertEqual(decoded["scalar"], 7)
        self.assertEqual(decoded["array"], [1.0, 2.0])
        self.assertEqual(decoded["objective"], str(ObjectiveLib.trace))


class TestDoeBuildHelpers(unittest.TestCase):
    """Coverage for small matrix helpers used by build/solve paths."""

    def test_symmetrize_lower_tri_helper(self):
        # Only the lower-triangular FIM is stored in several code paths; this
        # helper must mirror it to a full symmetric matrix without doubling the diagonal.
        m = np.array([[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]])
        got = DesignOfExperiments._symmetrize_lower_tri(m)
        expected = np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 5.0, 6.0]])
        self.assertTrue(np.allclose(got, expected, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
