# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________
import json
import warnings
from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)

from pyomo.common.errors import DeveloperError
import pyomo.common.unittest as unittest
from unittest.mock import patch

if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pyomo.DoE needs scipy and numpy to run tests")

from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.doe import InitializationMethod, _DoEResultsJSONEncoder
from pyomo.contrib.doe.tests.experiment_class_example_flags import (
    BadExperiment,
    RooneyBieglerExperimentFlag,
    RooneyBieglerMultiExperiment,
    RooneyBieglerMultiInputExperimentFlag,
)
from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
    RooneyBieglerExperiment,
)

if scipy_available:
    from pyomo.contrib.doe import DesignOfExperiments
    from pyomo.contrib.doe.doe import InitializationMethod, _DoEResultsJSONEncoder
    from pyomo.contrib.doe.tests.experiment_class_example_flags import (
        BadExperiment,
        RooneyBieglerExperimentFlag,
        RooneyBieglerMultiExperiment,
        RooneyBieglerMultiInputExperimentFlag,
    )
    from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
        RooneyBieglerExperiment,
    )

from pyomo.contrib.doe.examples.rooney_biegler_doe_example import run_rooney_biegler_doe
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()


class _DummyExperiment:
    def get_labeled_model(self, **kwargs):
        raise RuntimeError("Should not be called in argument-validation tests")


def get_rooney_biegler_experiment_flag():
    """Get a fresh RooneyBieglerExperimentFlag instance for testing.

    Creates a new experiment instance that supports the flag parameter
    for creating incomplete models for error testing.
    """
    data = pd.DataFrame(data=[[5, 15.6]], columns=['hour', 'y'])
    data_point = data.iloc[0]

    return RooneyBieglerExperimentFlag(
        data=data_point,
        theta={'asymptote': 15, 'rate_constant': 0.5},
        measure_error=0.1,
    )


def _make_ipopt_solver():
    solver = SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["halt_on_ampl_error"] = "yes"
    solver.options["max_iter"] = 3000
    return solver


def get_standard_args(experiment, fd_method, obj_used, flag):
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
    if flag is not None:
        args['get_labeled_model_args'] = {"flag": flag}
    args['_Cholesky_option'] = True
    args['_only_compute_fim_lower'] = True
    return args


@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestDoEErrors(unittest.TestCase):
    def _make_dummy_optimize_experiments_doe(self, n_experiments=1):
        return DesignOfExperiments(
            experiment=[_DummyExperiment() for _ in range(n_experiments)],
            objective_option="pseudo_trace",
        )

    def test_experiment_none_error(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

        with self.assertRaisesRegex(
            ValueError, "The 'experiment' parameter is required"
        ):
            # Experiment provided as None
            DoE_args = get_standard_args(None, fd_method, obj_used, flag_val)

            doe_obj = DesignOfExperiments(**DoE_args)

    def test_experiment_empty_list_error(self):
        with self.assertRaisesRegex(
            ValueError, "The 'experiment' list cannot be empty"
        ):
            DesignOfExperiments(experiment=[], objective_option="pseudo_trace")

    def test_doe_results_json_encoder_unsupported_object_raises(self):
        with self.assertRaises(TypeError):
            json.dumps({"x": object()}, cls=_DoEResultsJSONEncoder)

    def test_reactor_check_no_get_labeled_model(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

        experiment = BadExperiment()

        with self.assertRaisesRegex(
            ValueError, "Experiment at index .* must have a.*get_labeled_model"
        ):
            DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

            doe_obj = DesignOfExperiments(**DoE_args)

    def test_reactor_check_no_experiment_outputs(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"experiment_outputs".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_no_measurement_error(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = 2  # Value for faulty model build mode - 2: No meas error

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"measurement_error".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_no_experiment_inputs(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = 3  # Value for faulty model build mode - 3: No exp inputs/design vars

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"experiment_inputs".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_no_unknown_parameters(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = 4  # Value for faulty model build mode - 4: No unknown params

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Experiment model does not have suffix " + '"unknown_parameters".',
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_bad_prior_size(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        prior_FIM = np.ones((5, 5))

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)
        DoE_args['prior_FIM'] = prior_FIM

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "Shape of FIM provided should be n parameters by n parameters, "
            "or {} by {}, FIM provided has shape {} by {}".format(
                2, 2, prior_FIM.shape[0], prior_FIM.shape[1]
            ),
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_bad_prior_negative_eigenvalue(self):
        from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

        fd_method = "central"
        obj_used = "pseudo_trace"

        prior_FIM = -np.ones((2, 2))

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)
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
        obj_used = "pseudo_trace"

        prior_FIM = np.zeros((2, 2))
        prior_FIM[0, 1] = 1e-3

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)
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
        obj_used = "pseudo_trace"

        jac_init = np.ones((5, 5))

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)
        DoE_args['jac_initial'] = jac_init

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "Shape of Jacobian provided should be n experiment outputs "
            "by n parameters, or {} by {}, Jacobian provided has "
            "shape {} by {}".format(1, 2, jac_init.shape[0], jac_init.shape[1]),
        ):
            doe_obj.create_doe_model()

    def test_reactor_check_unbuilt_update_FIM(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        FIM_update = np.ones((2, 2))

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "``fim`` is not defined on the model provided. "
            "Please build the model first.",
        ):
            doe_obj.update_FIM_prior(FIM=FIM_update)

    def test_reactor_check_none_update_FIM(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        FIM_update = None

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "FIM input for update_FIM_prior must be a 2D, square numpy array.",
        ):
            doe_obj.update_FIM_prior(FIM=FIM_update)

    def test_reactor_check_results_file_name(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError, "``results_file`` must be either a Path object or a string."
        ):
            doe_obj.run_doe(results_file=int(15))

    def test_reactor_check_measurement_and_output_length_match(self):
        fd_method = "central"
        obj_used = "pseudo_trace"
        flag_val = (
            5  # Value for faulty model build mode - 5: Mismatch error and output length
        )

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag_val)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "Number of experiment outputs, {}, and length of measurement error, "
            "{}, do not match. Please check model labeling.".format(1, 2),
        ):
            doe_obj.create_doe_model()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_grid_search_des_range_inputs(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
    def test_reactor_premature_figure_drawing(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Results must be provided "
            "or the compute_FIM_full_factorial function must be run.",
        ):
            doe_obj.draw_factorial_figure()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_figure_drawing_no_des_var_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"hour": [1, 7, 2]}

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
    def test_reactor_figure_drawing_no_sens_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"hour": [1, 7, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError, "``sensitivity_design_variables`` must be included."
        ):
            doe_obj.draw_factorial_figure()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_figure_drawing_no_fixed_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"hour": [1, 7, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError, "``fixed_design_variables`` must be included."
        ):
            doe_obj.draw_factorial_figure(sensitivity_design_variables={"dummy": "var"})

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_figure_drawing_bad_fixed_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"hour": [1, 7, 2]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        with self.assertRaisesRegex(
            ValueError,
            "Fixed design variables do not all appear in the results object keys.",
        ):
            doe_obj.draw_factorial_figure(
                sensitivity_design_variables={"hour": 1},
                fixed_design_variables={"bad": "entry"},
            )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_reactor_figure_drawing_bad_sens_names(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"hour": [1, 7, 2]}

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
                fixed_design_variables={"hour": 1},
            )

    def test_reactor_check_get_FIM_without_FIM(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model provided does not have variable `fim`. Please make sure "
            "the model is built properly before calling `get_FIM`",
        ):
            doe_obj.get_FIM()

    def test_reactor_check_get_sens_mat_without_model(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            NotImplementedError, "Multiple experiment optimization not yet supported."
        ):
            doe_obj.run_multi_doe_sequential(N_exp=1)

    def test_multiple_exp_not_implemented_sim(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            NotImplementedError, "Multiple experiment optimization not yet supported."
        ):
            doe_obj.run_multi_doe_simultaneous(N_exp=1)

    def test_update_unknown_parameter_values_not_implemented_seq(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            NotImplementedError, "Updating unknown parameter values not yet supported."
        ):
            doe_obj.update_unknown_parameter_values()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_bad_FD_generate_scens(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
            doe_obj._generate_fd_scenario_blocks()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_bad_FD_seq_compute_FIM(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

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
        obj_used = "pseudo_trace"

        experiment = get_rooney_biegler_experiment_flag()

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            ValueError,
            "The method provided, {}, must be either `sequential` or `kaug`".format(
                "Bad Method"
            ),
        ):
            doe_obj.compute_FIM(method="Bad Method")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_compute_FIM_multi_experiment_parameter_value_mismatch(self):
        fd_method = "central"
        obj_used = "pseudo_trace"

        DoE_args = get_standard_args(
            RooneyBieglerMultiExperiment(hour=1.5, y=9.0), fd_method, obj_used, None
        )
        DoE_args["experiment"] = [
            RooneyBieglerMultiExperiment(
                hour=1.5, y=9.0, theta={'asymptote': 15, 'rate_constant': 0.5}
            ),
            RooneyBieglerMultiExperiment(
                hour=3.5, y=12.0, theta={'asymptote': 16, 'rate_constant': 0.5}
            ),
        ]
        doe_obj = DesignOfExperiments(**DoE_args)

        def _fake_sequential(*args, **kwargs):
            # This is only used if execution reaches the FIM solve call.
            doe_obj.seq_FIM = np.eye(2)

        with patch.object(doe_obj, "_sequential_FIM", side_effect=_fake_sequential):
            # The mismatch is detected before the second experiment solve,
            # when compute_FIM validates unknown parameter values.
            with self.assertRaisesRegex(
                ValueError, "must share the same unknown parameter values"
            ):
                doe_obj.compute_FIM(method="sequential")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    def test_invalid_trace_without_cholesky(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = run_rooney_biegler_doe()["experiment"]

        DoE_args = get_standard_args(experiment, fd_method, obj_used, flag=None)
        DoE_args['_Cholesky_option'] = False

        doe_obj = DesignOfExperiments(**DoE_args)
        doe_obj.create_doe_model()

        with self.assertRaisesRegex(
            ValueError,
            "objective_option='trace' currently only implemented with ``_Cholesky option=True``.",
        ):
            doe_obj.create_objective_function()

    def test_optimize_experiments_init_argument_validation_cases(self):
        # These argument checks all fail before any model build, so a single
        # table-driven test keeps the user-facing validation contracts aligned
        # without repeating the same dummy DoE setup for each branch.
        cases = [
            (
                "unsupported init_method",
                {"init_method": "bad"},
                ValueError,
                r"``init_method`` must be one of \[None, 'lhs'\], got 'bad'.",
            ),
            (
                "enum init_method still validates init_n_samples",
                {"init_method": InitializationMethod.lhs, "init_n_samples": 0},
                ValueError,
                r"``init_n_samples`` must be a positive integer, got 0.",
            ),
            (
                "non-positive init_n_samples",
                {"init_method": "lhs", "init_n_samples": 0},
                ValueError,
                r"``init_n_samples`` must be a positive integer, got 0.",
            ),
            (
                "non-integer init_n_samples",
                {"init_method": "lhs", "init_n_samples": 2.5},
                ValueError,
                r"``init_n_samples`` must be a positive integer, got 2.5.",
            ),
            (
                "init_parallel must be bool",
                {"init_method": "lhs", "init_parallel": 1},
                ValueError,
                r"``init_parallel`` must be a bool, got 1.",
            ),
            (
                "init_combo_parallel must be bool",
                {"init_method": "lhs", "init_combo_parallel": "yes"},
                ValueError,
                r"``init_combo_parallel`` must be a bool",
            ),
            (
                "init_n_workers must be positive integer",
                {"init_method": "lhs", "init_n_workers": 0},
                ValueError,
                r"``init_n_workers`` must be None or a positive integer",
            ),
            (
                "init_combo_chunk_size must be positive integer",
                {"init_method": "lhs", "init_combo_chunk_size": 0},
                ValueError,
                r"``init_combo_chunk_size`` must be a positive integer",
            ),
            (
                "init_combo_parallel_threshold must be positive integer",
                {"init_method": "lhs", "init_combo_parallel_threshold": 0},
                ValueError,
                r"``init_combo_parallel_threshold`` must be a positive integer",
            ),
            (
                "init_max_wall_clock_time must be positive",
                {"init_method": "lhs", "init_max_wall_clock_time": 0},
                ValueError,
                r"``init_max_wall_clock_time`` must be None or a positive number",
            ),
            (
                "init_max_wall_clock_time rejects nan",
                {"init_method": "lhs", "init_max_wall_clock_time": float("nan")},
                ValueError,
                r"``init_max_wall_clock_time`` must be None or a positive number",
            ),
            (
                "init_max_wall_clock_time rejects inf",
                {"init_method": "lhs", "init_max_wall_clock_time": float("inf")},
                ValueError,
                r"``init_max_wall_clock_time`` must be None or a positive number",
            ),
            (
                "init_seed must be integer",
                {
                    "n_exp": 2,
                    "init_method": "lhs",
                    "init_n_samples": 2,
                    "init_seed": 1.5,
                },
                ValueError,
                r"``init_seed`` must be None or an integer",
            ),
        ]

        for label, kwargs, exc_type, regex in cases:
            with self.subTest(case=label):
                doe_obj = self._make_dummy_optimize_experiments_doe()
                with self.assertRaisesRegex(exc_type, regex):
                    doe_obj.optimize_experiments(**kwargs)

    def test_optimize_experiments_lhs_requires_template_mode(self):
        # Tests that LHS initialization is disallowed in user-initialized multi-experiment mode.
        doe_obj = DesignOfExperiments(
            experiment=[_DummyExperiment(), _DummyExperiment()],
            objective_option="pseudo_trace",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"``init_method='lhs'`` is currently supported only in template mode",
        ):
            doe_obj.optimize_experiments(init_method="lhs")

    def test_optimize_experiments_lhs_requires_scipy(self):
        # Tests that LHS initialization requires scipy to be available.
        doe_obj = DesignOfExperiments(
            experiment=[_DummyExperiment()], objective_option="pseudo_trace"
        )
        with patch("pyomo.contrib.doe.doe.scipy_available", False):
            with self.assertRaisesRegex(
                ImportError, r"LHS initialization requires scipy"
            ):
                doe_obj.optimize_experiments(init_method="lhs")

    def test_optimize_experiments_general_argument_validation_cases(self):
        # These are the remaining lightweight API validations whose only
        # contract is the error raised for a bad user-facing kwarg/value pair.
        cases = [
            (
                "n_exp disallowed with multi-experiment list",
                2,
                {"n_exp": 2},
                ValueError,
                r"``n_exp`` must not be set when the experiment list contains more than one experiment",
            ),
            (
                "n_exp must be positive",
                1,
                {"n_exp": 0},
                ValueError,
                r"``n_exp`` must be a positive integer, got 0.",
            ),
            (
                "results_file must be str or Path",
                1,
                {"results_file": 5},
                ValueError,
                r"``results_file`` must be either a Path object or a string.",
            ),
            (
                "init_solver must have solve",
                1,
                {"init_solver": object()},
                ValueError,
                r"``init_solver`` must be None or a solver object with a 'solve' method.",
            ),
        ]

        for label, n_experiments, kwargs, exc_type, regex in cases:
            with self.subTest(case=label):
                doe_obj = self._make_dummy_optimize_experiments_doe(
                    n_experiments=n_experiments
                )
                with self.assertRaisesRegex(exc_type, regex):
                    doe_obj.optimize_experiments(**kwargs)


@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
@unittest.skipIf(not scipy_available, "scipy is not available")
@unittest.skipIf(not pandas_available, "pandas is not available")
class TestDoEErrorsRequiringSolver(unittest.TestCase):
    def _make_solver(self):
        return _make_ipopt_solver()

    def test_optimize_experiments_non_greybox_rejects_e_and_me_objectives(self):
        # E-opt and ME-opt require the greybox objective path in
        # optimize_experiments(); the standard algebraic multi-experiment build
        # should fail fast with a user-facing validation error instead.
        for objective_option in ("minimum_eigenvalue", "condition_number"):
            with self.subTest(objective=objective_option):
                doe_obj = DesignOfExperiments(
                    experiment=[RooneyBieglerMultiExperiment(hour=2.0, y=10.0)],
                    objective_option=objective_option,
                    step=1e-2,
                    solver=self._make_solver(),
                )

                with self.assertRaisesRegex(
                    ValueError,
                    rf"objective_option='{objective_option}' requires "
                    r"use_grey_box_objective=True\.",
                ):
                    doe_obj.optimize_experiments(n_exp=2)

    def test_optimize_experiments_requires_matching_unknown_parameter_values(self):
        # Tests that user-initialized multi-experiment mode rejects experiments
        # that linearize around different nominal theta values.
        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiExperiment(
                    hour=2.0, y=10.0, theta={'asymptote': 15.0, 'rate_constant': 0.5}
                ),
                RooneyBieglerMultiExperiment(
                    hour=3.0, y=11.0, theta={'asymptote': 15.0, 'rate_constant': 0.6}
                ),
            ],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )

        with self.assertRaisesRegex(
            ValueError, "must use the same nominal values for unknown_parameters"
        ):
            doe_obj.optimize_experiments()

    def test_optimize_experiments_requires_matching_unknown_parameter_labels(self):
        # Tests that user-initialized multi-experiment mode rejects experiments
        # whose unknown-parameter sets differ.
        class _DifferentUnknownParameterExperiment:
            def __init__(self, base_exp):
                self._base_exp = base_exp

            def get_labeled_model(self, **kwargs):
                m = self._base_exp.get_labeled_model(**kwargs)
                m.fake_theta = pyo.Var(initialize=1.0)
                m.fake_theta.fix()
                m.del_component(m.unknown_parameters)
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    [
                        (m.asymptote, pyo.value(m.asymptote)),
                        (m.fake_theta, pyo.value(m.fake_theta)),
                    ]
                )
                return m

        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiExperiment(hour=2.0, y=10.0),
                _DifferentUnknownParameterExperiment(
                    RooneyBieglerMultiExperiment(hour=3.0, y=11.0)
                ),
            ],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )

        with self.assertRaisesRegex(
            ValueError, "must define the same unknown_parameters in the same order"
        ):
            doe_obj.optimize_experiments()

    def test_optimize_experiments_requires_matching_unknown_parameter_order(self):
        # Tests that user-initialized multi-experiment mode rejects experiments
        # whose unknown parameters appear in a different order.
        class _ReorderedUnknownParameterExperiment:
            def __init__(self, base_exp):
                self._base_exp = base_exp

            def get_labeled_model(self, **kwargs):
                m = self._base_exp.get_labeled_model(**kwargs)
                m.del_component(m.unknown_parameters)
                m.unknown_parameters = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.unknown_parameters.update(
                    [
                        (m.rate_constant, pyo.value(m.rate_constant)),
                        (m.asymptote, pyo.value(m.asymptote)),
                    ]
                )
                return m

        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiExperiment(hour=2.0, y=10.0),
                _ReorderedUnknownParameterExperiment(
                    RooneyBieglerMultiExperiment(hour=3.0, y=11.0)
                ),
            ],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )

        with self.assertRaisesRegex(
            ValueError, "must define the same unknown_parameters in the same order"
        ):
            doe_obj.optimize_experiments()

    def test_optimize_experiments_sym_break_var_must_be_input(self):
        # Tests that symmetry-breaking marker variables must also be experiment inputs.
        class _BadSymBreakExperiment:
            def __init__(self, base_exp):
                self._base_exp = base_exp

            def get_labeled_model(self, **kwargs):
                m = self._base_exp.get_labeled_model(**kwargs)
                m.sym_break_cons = pyo.Suffix(direction=pyo.Suffix.LOCAL)
                m.sym_break_cons[next(iter(m.unknown_parameters.keys()))] = None
                return m

        exp = _BadSymBreakExperiment(RooneyBieglerMultiExperiment(hour=2.0, y=10.0))
        doe_obj = DesignOfExperiments(
            experiment=[exp],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )

        with self.assertRaisesRegex(
            ValueError, "sym_break_cons.*must also be an experiment input variable"
        ):
            doe_obj.optimize_experiments(n_exp=2)

    def test_optimize_experiments_symmetry_mapping_failure_raises(self):
        # Tests that failure to map symmetry variable across experiment blocks raises.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0)],
            objective_option="pseudo_trace",
            step=1e-2,
        )
        probe_model = doe_obj.experiment_list[0].get_labeled_model(
            **doe_obj.get_labeled_model_args
        )
        sym_var_name = next(iter(probe_model.experiment_inputs.keys())).local_name
        original_find = pyo.ComponentUID.find_component_on

        def _fail_only_symmetry_mapping(cuid, block):
            if (
                sym_var_name in str(cuid)
                and hasattr(block, "experiment_inputs")
                and block.index() == 0
            ):
                return None
            return original_find(cuid, block)

        with patch(
            "pyomo.contrib.doe.doe.pyo.ComponentUID.find_component_on",
            autospec=True,
            side_effect=_fail_only_symmetry_mapping,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Failed to map symmetry breaking variable"
            ):
                doe_obj.optimize_experiments(n_exp=2)

    def test_optimize_experiments_symmetry_breaking_default_variable_warning(self):
        # Tests that missing explicit symmetry marker triggers warning and default choice.
        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiInputExperimentFlag(hour=2.0, sym_break_flag=0),
                RooneyBieglerMultiInputExperimentFlag(hour=4.0, sym_break_flag=0),
            ],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as cm:
            doe_obj.optimize_experiments()
        self.assertTrue(
            any("No symmetry breaking variable specified" in msg for msg in cm.output)
        )
        self.assertTrue(
            hasattr(doe_obj.model.param_scenario_blocks[0], "symmetry_breaking_s0_exp1")
        )

    def test_optimize_experiments_symmetry_breaking_multiple_markers_warning(self):
        # Tests that multiple symmetry markers trigger an ambiguity warning.
        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiInputExperimentFlag(hour=2.0, sym_break_flag=2),
                RooneyBieglerMultiInputExperimentFlag(hour=4.0, sym_break_flag=2),
            ],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as cm:
            doe_obj.optimize_experiments()
        self.assertTrue(
            any(
                "Multiple variables marked in sym_break_cons" in msg
                for msg in cm.output
            )
        )

    def test_lhs_initialization_large_space_emits_warnings(self):
        # Tests that very large LHS candidate/combo spaces emit user-facing warnings.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0, y=10.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as log_cm:
            with warnings.catch_warnings(record=True) as warn_cm:
                warnings.simplefilter("always")
                with patch(
                    "pyomo.contrib.doe.doe._combinations", return_value=iter([(0, 1)])
                ):
                    with patch.object(
                        doe_obj,
                        "_compute_fim_at_point_no_prior",
                        return_value=np.eye(2),
                    ):
                        doe_obj.optimize_experiments(
                            n_exp=2,
                            init_method="lhs",
                            init_n_samples=10001,
                            init_seed=11,
                        )

        self.assertTrue(
            any("candidate experiment designs" in str(w.message) for w in warn_cm)
        )
        self.assertTrue(any("combinations to evaluate" in msg for msg in log_cm.output))

    def test_lhs_combo_parallel_requested_but_not_used_warns(self):
        # Tests that combo-parallel requests warn when thresholds force serial scoring.
        doe_obj = DesignOfExperiments(
            experiment=[RooneyBieglerMultiExperiment(hour=2.0, y=10.0)],
            objective_option="pseudo_trace",
            step=1e-2,
            solver=self._make_solver(),
        )
        with patch.object(
            doe_obj, "_compute_fim_at_point_no_prior", return_value=np.eye(2)
        ):
            with self.assertLogs("pyomo.contrib.doe.doe", level="WARNING") as cm:
                doe_obj.optimize_experiments(
                    n_exp=2,
                    init_method="lhs",
                    init_n_samples=2,
                    init_seed=11,
                    init_combo_parallel=True,
                    init_n_workers=2,
                    init_combo_parallel_threshold=10_000,
                )

        self.assertTrue(
            any(
                "lhs_combo_parallel=True" in msg and "running serially" in msg
                for msg in cm.output
            )
        )

    def test_lhs_missing_bounds_error_message(self):
        # Tests that LHS initialization fails fast when experiment inputs lack bounds.
        doe_obj = DesignOfExperiments(
            experiment=[
                RooneyBieglerMultiExperiment(hour=2.0, hour_bounds=(None, 10.0))
            ],
            objective_option="pseudo_trace",
        )
        with self.assertRaisesRegex(
            ValueError,
            r"LHS initialization requires explicit lower and upper bounds on all experiment input variables",
        ):
            doe_obj.optimize_experiments(init_method="lhs", init_n_samples=2)


if __name__ == "__main__":
    unittest.main()
