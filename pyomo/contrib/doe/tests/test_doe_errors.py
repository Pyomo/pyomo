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

from pyomo.common.errors import DeveloperError
from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

if scipy_available:
    from pyomo.contrib.doe import DesignOfExperiments
    from pyomo.contrib.doe.tests.experiment_class_example_flags import (
        BadExperiment,
        FullReactorExperiment,
    )

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)
data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}


def get_standard_args(experiment, fd_method, obj_used, flag):
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
    args['get_labeled_model_args'] = {"flag": flag}
    args['_Cholesky_option'] = True
    args['_only_compute_fim_lower'] = True
    return args


@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
class TestReactorExampleErrors(unittest.TestCase):
    def test_experiment_none_error(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

        with self.assertRaisesRegex(
            ValueError, "Experiment object must be provided to perform DoE."
        ):
            # Experiment provided as None
            DoE_args = get_standard_args(None, fd_method, obj_used, flag_val)

            doe_obj = DesignOfExperiments(**DoE_args)

    def test_reactor_check_no_get_labeled_model(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

        experiment = BadExperiment()

        with self.assertRaisesRegex(
            ValueError,
            "The experiment object must have a ``get_labeled_model`` function",
        ):
            DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

            doe_obj = DesignOfExperiments(**DoE_args)

    def test_reactor_check_no_experiment_outputs(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"experiment_outputs".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_no_measurement_error(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 2  # Value for faulty model build mode - 2: No meas error

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"measurement_error".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_no_experiment_inputs(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 3  # Value for faulty model build mode - 3: No exp inputs/design vars

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"experiment_inputs".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_no_unknown_parameters(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 4  # Value for faulty model build mode - 4: No unknown params

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"unknown_parameters".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_bad_prior_size(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: full model

        prior_FIM = np.ones((5, 5))

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)
        DoE_args['prior_FIM'] = prior_FIM

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "Shape of FIM provided should be n parameters by n parameters, "
            "or {} by {}, FIM provided has shape {} by {}".format(
                4, 4, prior_FIM.shape[0], prior_FIM.shape[1]
            ),
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_bad_prior_negative_eigenvalue(self):
        from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: full model

        prior_FIM = -np.ones((4, 4))

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)
        DoE_args['prior_FIM'] = prior_FIM

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            r"FIM provided is not positive definite. It has one or more "
            r"negative eigenvalue\(s\) less than -{:.1e}".format(
                _SMALL_TOLERANCE_DEFINITENESS
            ),
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_bad_prior_not_symmetric(self):
        from pyomo.contrib.doe.utils import _SMALL_TOLERANCE_SYMMETRY

        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: full model

        prior_FIM = np.zeros((4, 4))
        prior_FIM[0, 1] = 1e-3

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)
        DoE_args['prior_FIM'] = prior_FIM

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "FIM provided is not symmetric using absolute tolerance {}".format(
                _SMALL_TOLERANCE_SYMMETRY
            ),
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_bad_jacobian_init_size(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: full model

        jac_init = np.ones((5, 5))

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)
        DoE_args['jac_initial'] = jac_init

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "Shape of Jacobian provided should be n experiment outputs "
            "by n parameters, or {} by {}, Jacobian provided has "
            "shape {} by {}".format(27, 4, jac_init.shape[0], jac_init.shape[1]),
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_unbuilt_update_FIM(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: full model

        FIM_update = np.ones((4, 4))

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "``fim`` is not defined on the model provided. "
            "Please build the model first.",
        ):
            doe_obj.update_FIM_prior(FIM=FIM_update)

    def test_reactor_check_none_update_FIM(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: full model

        FIM_update = None

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "FIM input for update_FIM_prior must be a 2D, square numpy array.",
        ):
            doe_obj.update_FIM_prior(FIM=FIM_update)

    def test_reactor_check_results_file_name(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 0  # Value for faulty model build mode - 0: Full model

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError, "``results_file`` must be either a Path object or a string."
        ):
            doe_obj.run_doe(results_file=int(15))

    def test_reactor_check_measurement_and_output_length_match(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            5  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "Number of experiment outputs, {}, and length of measurement error, "
            "{}, do not match. Please check model labeling.".format(27, 1),
        ):
            doe_obj.create_doe_model()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_grid_search_des_range_inputs(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"not": [1, 5, 3], "correct": [300, 700, 3]}

        with self.assertRaisesRegex(
            ValueError,
            "Design ranges keys must be a subset of experimental design names.",
        ):
            doe_obj.compute_FIM_full_factorial(
                design_ranges=design_ranges, method="sequential"
            )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_premature_figure_drawing(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Results must be provided "
            "or the compute_FIM_full_factorial function must be run.",
        ):
            doe_obj.draw_factorial_figure()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_figure_drawing_no_des_var_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError,
            "If results object is provided, you must "
            "include all the design variable names.",
        ):
            doe_obj.draw_factorial_figure(results=doe_obj.fim_factorial_results)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_figure_drawing_no_sens_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError, "``sensitivity_design_variables`` must be included."
        ):
            doe_obj.draw_factorial_figure()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_figure_drawing_no_fixed_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError, "``fixed_design_variables`` must be included."
        ):
            doe_obj.draw_factorial_figure(sensitivity_design_variables={"dummy": "var"})

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_figure_drawing_bad_fixed_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError,
            "Fixed design variables do not all appear in the results object keys.",
        ):
            doe_obj.draw_factorial_figure(
                sensitivity_design_variables={"CA[0]": 1},
                fixed_design_variables={"bad": "entry"},
            )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_figure_drawing_bad_sens_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=0)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError,
            "Sensitivity design variables do not all appear "
            "in the results object keys.",
        ):
            doe_obj.draw_factorial_figure(
                sensitivity_design_variables={"bad": "entry"},
                fixed_design_variables={"CA[0]": 1},
            )

    def test_reactor_check_get_FIM_without_FIM(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have variable `fim`. Please make sure "
            "the model is built properly before calling `get_FIM`",
        ):
            doe_obj.get_FIM()

    def test_reactor_check_get_sens_mat_without_model(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have variable `sensitivity_jacobian`. "
            "Please make sure the model is built properly before calling "
            "`get_sensitivity_matrix`",
        ):
            doe_obj.get_sensitivity_matrix()

    def test_reactor_check_get_exp_inputs_without_model(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have expected structure. "
            "Please make sure model is built properly before "
            "calling `get_experiment_input_values`",
        ):
            doe_obj.get_experiment_input_values()

    def test_reactor_check_get_exp_outputs_without_model(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have expected structure. Please make "
            "sure model is built properly before calling "
            "`get_experiment_output_values`",
        ):
            doe_obj.get_experiment_output_values()

    def test_reactor_check_get_unknown_params_without_model(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have expected structure. Please make "
            "sure model is built properly before calling "
            "`get_unknown_parameter_values`",
        ):
            doe_obj.get_unknown_parameter_values()

    def test_reactor_check_get_meas_error_without_model(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have expected structure. Please make "
            "sure model is built properly before calling "
            "`get_measurement_error_values`",
        ):
            doe_obj.get_measurement_error_values()

    def test_multiple_exp_not_implemented_seq(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            NotImplementedError, "Multiple experiment optimization not yet supported."
        ):
            doe_obj.run_multi_doe_sequential(N_exp=1)

    def test_multiple_exp_not_implemented_sim(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            NotImplementedError, "Multiple experiment optimization not yet supported."
        ):
            doe_obj.run_multi_doe_simultaneous(N_exp=1)

    def test_update_unknown_parameter_values_not_implemented_seq(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            NotImplementedError, "Updating unknown parameter values not yet supported."
        ):
            doe_obj.update_unknown_parameter_values()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_bad_FD_generate_scens(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            DeveloperError,
            "Internal Pyomo implementation error:\n"
            "        "
            "'Finite difference option not recognized. Please contact the\n"
            "        "
            "developers as you should not see this error.'\n"
            "    "
            "Please report this to the Pyomo Developers.",
        ):
            doe_obj.fd_formula = "bad things"
            doe_obj._generate_scenario_blocks()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_bad_FD_seq_compute_FIM(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            DeveloperError,
            "Internal Pyomo implementation error:\n"
            "        "
            "'Finite difference option not recognized. Please contact the\n"
            "        "
            "developers as you should not see this error.'\n"
            "    "
            "Please report this to the Pyomo Developers.",
        ):
            doe_obj.fd_formula = "bad things"
            doe_obj.compute_FIM(method="sequential")

    def test_bad_objective(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            DeveloperError,
            "Internal Pyomo implementation error:\n"
            "        "
            "'Objective option not recognized. Please contact the developers as\n"
            "        "
            "you should not see this error.'\n"
            "    "
            "Please report this to the Pyomo Developers.",
        ):
            doe_obj.objective_option = "bad things"
            doe_obj.create_objective_function()

    def test_no_model_for_objective(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have variable `fim`. Please make "
            "sure the model is built properly before creating the objective.",
        ):
            doe_obj.create_objective_function()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_bad_compute_FIM_option(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = (
            0  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "The method provided, {}, must be either `sequential` or `kaug`".format(
                "Bad Method"
            ),
        ):
            doe_obj.compute_FIM(method="Bad Method")


if __name__ == "__main__":
    unittest.main()
