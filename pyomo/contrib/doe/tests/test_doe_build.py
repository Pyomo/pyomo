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
import json
import os.path

from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)

from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

if scipy_available:
    from pyomo.contrib.doe import DesignOfExperiments
    from pyomo.contrib.doe.examples.reactor_example import (
        ReactorExperiment as FullReactorExperiment,
    )

import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)

data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}


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
    sigma_inv = [1 / v for k, v in model.scenario_blocks[0].measurement_error.items()]
    param_vals = np.array(
        [[v for k, v in model.scenario_blocks[0].unknown_parameters.items()]]
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
    args['experiment'] = experiment
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
class TestReactorExampleBuild(unittest.TestCase):
    def test_reactor_fd_central_check_fd_eqns(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the parameter values are correct
        for s in model.scenarios:
            param = model.parameter_scenarios[s]

            diff = (-1) ** s * doe_obj.step

            param_val = pyo.value(
                pyo.ComponentUID(param).find_component_on(model.scenario_blocks[s])
            )

            param_val_from_step = model.scenario_blocks[0].unknown_parameters[
                pyo.ComponentUID(param).find_component_on(model.scenario_blocks[0])
            ] * (1 + diff)

            for k, v in model.scenario_blocks[s].unknown_parameters.items():
                if pyo.ComponentUID(
                    k, context=model.scenario_blocks[s]
                ) == pyo.ComponentUID(param):
                    continue

                other_param_val = pyo.value(k)
                self.assertAlmostEqual(other_param_val, v)

            self.assertAlmostEqual(param_val, param_val_from_step)

    def test_reactor_fd_backward_check_fd_eqns(self):
        fd_method = "backward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

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
                    pyo.ComponentUID(param).find_component_on(model.scenario_blocks[s])
                )

                param_val_from_step = model.scenario_blocks[0].unknown_parameters[
                    pyo.ComponentUID(param).find_component_on(model.scenario_blocks[0])
                ] * (1 + diff)
                self.assertAlmostEqual(param_val, param_val_from_step)

            for k, v in model.scenario_blocks[s].unknown_parameters.items():
                if (s != 0) and pyo.ComponentUID(
                    k, context=model.scenario_blocks[s]
                ) == pyo.ComponentUID(param):
                    continue

                other_param_val = pyo.value(k)
                self.assertAlmostEqual(other_param_val, v)

    def test_reactor_fd_forward_check_fd_eqns(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

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
                    pyo.ComponentUID(param).find_component_on(model.scenario_blocks[s])
                )

                param_val_from_step = model.scenario_blocks[0].unknown_parameters[
                    pyo.ComponentUID(param).find_component_on(model.scenario_blocks[0])
                ] * (1 + diff)
                self.assertAlmostEqual(param_val, param_val_from_step)

            for k, v in model.scenario_blocks[s].unknown_parameters.items():
                if (s != 0) and pyo.ComponentUID(
                    k, context=model.scenario_blocks[s]
                ) == pyo.ComponentUID(param):
                    continue

                other_param_val = pyo.value(k)
                self.assertAlmostEqual(other_param_val, v)

    def test_reactor_fd_central_design_fixing(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the design fixing constraints are generated
        design_vars = [k for k, v in model.scenario_blocks[0].experiment_inputs.items()]

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

    def test_reactor_fd_backward_design_fixing(self):
        fd_method = "backward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the design fixing constraints are generated
        design_vars = [k for k, v in model.scenario_blocks[0].experiment_inputs.items()]

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

    def test_reactor_fd_forward_design_fixing(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.create_doe_model()

        model = doe_obj.model

        # Check that the design fixing constraints are generated
        design_vars = [k for k, v in model.scenario_blocks[0].experiment_inputs.items()]

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

    def test_reactor_check_user_initialization(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_prior = np.ones((4, 4))
        FIM_initial = np.eye(4) + FIM_prior
        JAC_initial = np.ones((27, 4)) * 2

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
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_update = np.ones((4, 4)) * 10

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
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

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
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

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
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

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
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

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
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj._generate_scenario_blocks()

        for i in doe_obj.model.parameter_scenarios:
            self.assertTrue(
                doe_obj.model.find_component("scenario_blocks[" + str(i) + "]")
            )

    def test_reactor_update_suffix_items(self):
        """Test the reactor example with updating suffix items."""
        from pyomo.contrib.doe.examples.update_suffix_doe_example import main

        # Run the reactor update suffix items example
        suffix_obj, _, new_vals = main()

        # Check that the suffix object has been updated correctly
        for i, v in enumerate(suffix_obj.values()):
            self.assertAlmostEqual(v, new_vals[i], places=6)


if __name__ == "__main__":
    unittest.main()
