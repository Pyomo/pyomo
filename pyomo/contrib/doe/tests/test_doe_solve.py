from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
    scipy_available,
)

from pyomo.contrib.doe.tests.experiment_class_example import *
from pyomo.contrib.doe.tests.experiment_class_example_flags import (
    FullReactorExperimentBad,
)
from pyomo.contrib.doe import *
from pyomo.contrib.doe.utils import *

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.opt import SolverFactory

from pathlib import Path

import logging

ipopt_available = SolverFactory("ipopt").available()
k_aug_available = SolverFactory('k_aug', solver_io='nl', validate=False)

DATA_DIR = Path(__file__).parent
file_path = DATA_DIR / "result.json"

f = open(file_path)
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


class TestReactorExampleSolving(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_central_solve(self):
        fd_method = "central"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.run_doe()

        # Assert model solves
        assert doe_obj.results["Solver Status"] == "ok"

        # Assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        assert np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_forward_solve(self):
        fd_method = "forward"
        obj_used = "zero"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.run_doe()

        assert doe_obj.results["Solver Status"] == "ok"

        # Assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        assert np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_backward_solve(self):
        fd_method = "backward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.run_doe()

        assert doe_obj.results["Solver Status"] == "ok"

        # Assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Trace is used, no comparison for FIM and L.T @ L

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        assert np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q))

    # TODO: Fix determinant objective code, something is awry
    #       Should only be using Cholesky=True
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_obj_det_solve(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=False,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=False,
            _only_compute_fim_lower=False,
        )

        doe_obj.run_doe()

        assert doe_obj.results['Solver Status'] == "ok"

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_obj_cholesky_solve(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.run_doe()

        assert doe_obj.results["Solver Status"] == "ok"

        # Assert that Q, F, and L are the same.
        FIM, Q, L, sigma_inv = get_FIM_Q_L(doe_obj=doe_obj)

        # Since Cholesky is used, there is comparison for FIM and L.T @ L
        assert np.all(np.isclose(FIM, L @ L.T))

        # Make sure FIM and Q.T @ sigma_inv @ Q are close (alternate definition of FIM)
        assert np.all(np.isclose(FIM, Q.T @ sigma_inv @ Q))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_compute_FIM_seq_centr(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.compute_FIM(method="sequential")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_compute_FIM_seq_forward(self):
        fd_method = "forward"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.compute_FIM(method="sequential")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not scipy_available, "Scipy is not available")
    @unittest.skipIf(
        not k_aug_available.available(False), "The 'k_aug' command is not available"
    )
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_compute_FIM_kaug(self):
        fd_method = "forward"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.compute_FIM(method="kaug")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_compute_FIM_seq_backward(self):
        fd_method = "backward"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.compute_FIM(method="sequential")

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_grid_search(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        design_ranges = {"CA[0]": [1, 5, 3], "T[0]": [300, 700, 3]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check to make sure the lengths of the inputs in results object are indeed correct
        CA_vals = doe_obj.fim_factorial_results["CA[0]"]
        T_vals = doe_obj.fim_factorial_results["T[0]"]

        # Assert length is correct
        assert (len(CA_vals) == 9) and (len(T_vals) == 9)
        assert (len(set(CA_vals)) == 3) and (len(set(T_vals)) == 3)

        # Assert unique values are correct
        assert (set(CA_vals).issuperset(set([1, 3, 5]))) and (
            set(T_vals).issuperset(set([300, 500, 700]))
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_rescale_FIM(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        # With parameter scaling
        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        # Without parameter scaling
        doe_obj2 = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=False,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

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
        assert np.all(np.isclose(FIM2, resc_FIM))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_solve_bad_model(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperimentBad(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Model from experiment did not solve appropriately. Make sure the model is well-posed.",
        ):
            doe_obj.run_doe()

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not pandas_available, "pandas is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_grid_search_bad_model(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperimentBad(data_ex, 10, 3)

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=None,
            jac_initial=None,
            fim_initial=None,
            L_initial=None,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
            logger_level=logging.ERROR,
        )

        design_ranges = {"CA[0]": [1, 5, 3], "T[0]": [300, 700, 3]}

        doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )

        # Check to make sure the lengths of the inputs in results object are indeed correct
        CA_vals = doe_obj.fim_factorial_results["CA[0]"]
        T_vals = doe_obj.fim_factorial_results["T[0]"]

        # Assert length is correct
        assert (len(CA_vals) == 9) and (len(T_vals) == 9)
        assert (len(set(CA_vals)) == 3) and (len(set(T_vals)) == 3)

        # Assert unique values are correct
        assert (set(CA_vals).issuperset(set([1, 3, 5]))) and (
            set(T_vals).issuperset(set([300, 500, 700]))
        )


if __name__ == "__main__":
    unittest.main()
