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
import logging
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
    from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
    from pyomo.contrib.doe.examples.reactor_example import (
        ReactorExperiment as FullReactorExperiment,
        run_reactor_doe,
    )
    from pyomo.contrib.doe.tests.experiment_class_example_flags import (
        FullReactorExperimentBad,
    )
from pyomo.contrib.doe.utils import rescale_FIM

import pyomo.environ as pyo

from pyomo.opt import SolverFactory


ipopt_available = SolverFactory("ipopt").available()
k_aug_available = SolverFactory("k_aug", solver_io="nl", validate=False)

currdir = this_file_dir()
file_path = os.path.join(currdir, "..", "examples", "result.json")

with open(file_path) as f:
    data_ex = json.load(f)
data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}


def get_FIM_Q_L(doe_obj=None):
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

    for i in range(n_param):
        for j in range(n_param):
            if j < i:
                FIM_vals_np[j, i] = FIM_vals_np[i, j]

    L_vals_np = np.array(L_vals).reshape((n_param, n_param))
    Q_vals_np = np.array(Q_vals).reshape((n_y, n_param))

    sigma_inv_np = np.zeros((n_y, n_y))

    for ind, v in enumerate(sigma_inv):
        sigma_inv_np[ind, ind] = v

    return FIM_vals_np, Q_vals_np, L_vals_np, sigma_inv_np


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
class TestReactorExampleSolving(unittest.TestCase):
    def test_reactor_fd_central_solve(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        # assert model solves
        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    def test_reactor_fd_forward_solve(self):
        fd_method = "forward"
        obj_used = "zero"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    def test_reactor_fd_backward_solve(self):
        fd_method = "backward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    def test_reactor_obj_det_solve(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["scale_nominal_param_value"] = (
            False  # Vanilla determinant solve needs this
        )
        DoE_args["_Cholesky_option"] = False
        DoE_args["_only_compute_fim_lower"] = False

        doe_obj = DesignOfExperiments(**DoE_args)

        # Increase numerical performance by adding a prior
        prior_FIM = doe_obj.compute_FIM()
        doe_obj.prior_FIM = prior_FIM

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

    def test_reactor_obj_cholesky_solve(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Cholesky is used, there is comparison for FIM and L.T @ L
        self.assertTrue(np.all(np.isclose(FIM, L @ L.T)))

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    def test_reactor_obj_cholesky_solve_bad_prior(self):
        from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_DEFINITENESS

        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        # Specify a prior that is slightly negative definite
        # Because it is less than the tolerance, it should be
        # adjusted to be positive definite
        # No error should be thrown
        DoE_args["prior_FIM"] = -(_SMALL_TOLERANCE_DEFINITENESS / 100) * np.eye(4)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.run_doe()

        self.assertEqual(doe_obj.results["Solver Status"], "ok")

        # assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Cholesky is used, there is comparison for FIM and L.T @ L
        self.assertTrue(np.all(np.isclose(FIM, L @ L.T)))

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        self.assertTrue(np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q)))

    # This test ensure that compute FIM runs without error using the
    # `sequential` option with central finite differences
    def test_compute_FIM_seq_centr(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

    # This test ensure that compute FIM runs without error using the
    # `sequential` option with forward finite differences
    def test_compute_FIM_seq_forward(self):
        fd_method = "forward"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

    # This test ensure that compute FIM runs without error using the
    # `kaug` option. kaug computes the FIM directly so no finite difference
    # scheme is needed.
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    @unittest.skipIf(
        not k_aug_available.available(False), "The 'k_aug' command is not available"
    )
    def test_compute_FIM_kaug(self):
        fd_method = "forward"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="kaug")

    # This test ensure that compute FIM runs without error using the
    # `sequential` option with backward finite differences
    def test_compute_FIM_seq_backward(self):
        fd_method = "backward"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        doe_obj.compute_FIM(method="sequential")

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_grid_search(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 3], "T[0]": [300, 700, 3]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check to make sure the lengths of the inputs
        # in results object are indeed correct
        CA_vals = doe_obj.fim_factorial_results["CA[0]"]
        T_vals = doe_obj.fim_factorial_results["T[0]"]

        # assert length is correct
        self.assertTrue((len(CA_vals) == 9) and (len(T_vals) == 9))
        self.assertTrue((len(set(CA_vals)) == 3) and (len(set(T_vals)) == 3))

        # assert unique values are correct
        self.assertTrue(
            (set(CA_vals).issuperset(set([1, 3, 5])))
            and (set(T_vals).issuperset(set([300, 500, 700])))
        )

    def test_rescale_FIM(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        # With parameter scaling
        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        # Without parameter scaling
        DoE_args2 = get_standard_args(experiment, fd_method, obj_used)
        DoE_args2["scale_nominal_param_value"] = False

        doe_obj2 = DesignOfExperiments(**DoE_args2)
        # Run both problems
        doe_obj.run_doe()
        doe_obj2.run_doe()

        # Extract FIM values
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)
        FIM2, Q2, L2, sigma_inv2 = get_FIM_Q_L(doe_obj=doe_obj2)

        # Get rescaled FIM from the scaled version
        param_vals = np.array(
            [
                [
                    v
                    for k, v in doe_obj.model.scenario_blocks[
                        0
                    ].unknown_parameters.items()
                ]
            ]
        )

        resc_FIM = rescale_FIM(FIM, param_vals)

        # Compare scaled and rescaled values
        self.assertTrue(np.all(np.isclose(FIM2, resc_FIM)))

    def test_reactor_solve_bad_model(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperimentBad(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)

        doe_obj = DesignOfExperiments(**DoE_args)

        with self.assertRaisesRegex(
            RuntimeError,
            "Model from experiment did not solve appropriately. "
            "Make sure the model is well-posed.",
        ):
            doe_obj.run_doe()

    @unittest.skipIf(not pandas_available, "pandas is not available")
    def test_reactor_grid_search_bad_model(self):
        fd_method = "central"
        obj_used = "determinant"

        experiment = FullReactorExperimentBad(data_ex, 10, 3)

        DoE_args = get_standard_args(experiment, fd_method, obj_used)
        DoE_args["logger_level"] = logging.ERROR

        doe_obj = DesignOfExperiments(**DoE_args)

        design_ranges = {"CA[0]": [1, 5, 3], "T[0]": [300, 700, 3]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check to make sure the lengths of the inputs in results object are indeed correct
        CA_vals = doe_obj.fim_factorial_results["CA[0]"]
        T_vals = doe_obj.fim_factorial_results["T[0]"]

        # assert length is correct
        self.assertTrue((len(CA_vals) == 9) and (len(T_vals) == 9))
        self.assertTrue((len(set(CA_vals)) == 3) and (len(set(T_vals)) == 3))

        # assert unique values are correct
        self.assertTrue(
            (set(CA_vals).issuperset(set([1, 3, 5])))
            and (set(T_vals).issuperset(set([300, 500, 700])))
        )


@unittest.skipIf(not ipopt_available, "The 'ipopt' solver is not available")
@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestDoe(unittest.TestCase):
    def test_doe_full_factorial(self):
        log10_D_opt_expected = [
            3.7734377852467524,
            5.137792359070963,
            5.182167857710023,
            6.546522431509408,
        ]

        log10_A_opt_expected = [
            3.5935726800929695,
            3.6133186151486933,
            3.945755198204365,
            3.9655011332598367,
        ]

        log10_E_opt_expected = [
            -1.7201873126109162,
            -0.691340497355524,
            -1.3680047944877138,
            -0.3391579792516522,
        ]

        log10_ME_opt_expected = [
            5.221185311075697,
            4.244741560076784,
            5.221185311062606,
            4.244741560083524,
        ]

        eigval_min_expected = [
            0.019046390638130666,
            0.20354456134677426,
            0.04285437893696232,
            0.45797526302234304,
        ]

        eigval_max_expected = [
            3169.552855492114,
            3576.0292523637977,
            7131.493924857995,
            8046.0658178139165,
        ]

        det_FIM_expected = [
            5935.233170586055,
            137338.51875774842,
            152113.5345070818,
            3519836.021699428,
        ]

        trace_FIM_expected = [
            3922.5878617108597,
            4105.051549241871,
            8825.822688850109,
            9236.36598578955,
        ]
        ff = run_reactor_doe(
            n_points_for_design=2,
            compute_FIM_full_factorial=False,
            plot_factorial_results=False,
            save_plots=False,
            run_optimal_doe=False,
        )
        ff.compute_FIM_full_factorial(
            design_ranges={"CA[0]": [1, 1.5, 2], "T[0]": [350, 400, 2]}
        )

        ff_results = ff.fim_factorial_results

        self.assertStructuredAlmostEqual(
            ff_results["log10 D-opt"], log10_D_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["log10 A-opt"], log10_A_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["log10 E-opt"], log10_E_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["log10 ME-opt"], log10_ME_opt_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["eigval_min"], eigval_min_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["eigval_max"], eigval_max_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["det_FIM"], det_FIM_expected, abstol=1e-4
        )
        self.assertStructuredAlmostEqual(
            ff_results["trace_FIM"], trace_FIM_expected, abstol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
