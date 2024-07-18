from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)

from pyomo.contrib.doe.tests.experiment_class_example import *
from pyomo.contrib.doe import *


import pyomo.common.unittest as unittest

from pyomo.opt import SolverFactory

from pathlib import Path

ipopt_available = SolverFactory("ipopt").available()

DATA_DIR = Path(__file__).parent
file_path = DATA_DIR / "result.json"

f = open(file_path)
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


class TestReactorExampleBuild(unittest.TestCase):
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_central_check_fd_eqns(self):
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
                name_ind = k.name.split(".").index("scenario_blocks[" + str(s) + "]")

                if ".".join(k.name.split(".")[name_ind + 1 :]) == param.name:
                    continue

                other_param_val = pyo.value(k)
                assert np.isclose(other_param_val, v)

            assert np.isclose(param_val, param_val_from_step)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_backward_check_fd_eqns(self):
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
                assert np.isclose(param_val, param_val_from_step)

            for k, v in model.scenario_blocks[s].unknown_parameters.items():
                name_ind = k.name.split(".").index("scenario_blocks[" + str(s) + "]")

                if (
                    not (s == 0)
                    and ".".join(k.name.split(".")[name_ind + 1 :]) == param.name
                ):
                    continue

                other_param_val = pyo.value(k)
                assert np.isclose(other_param_val, v)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_forward_check_fd_eqns(self):
        fd_method = "forward"
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
                assert np.isclose(param_val, param_val_from_step)

            for k, v in model.scenario_blocks[s].unknown_parameters.items():
                name_ind = k.name.split(".").index("scenario_blocks[" + str(s) + "]")

                if (
                    not (s == 0)
                    and ".".join(k.name.split(".")[name_ind + 1 :]) == param.name
                ):
                    continue

                other_param_val = pyo.value(k)
                assert np.isclose(other_param_val, v)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_central_design_fixing(self):
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
            assert hasattr(model, con_name)

            # Ensure that each set of constraints has all blocks pairs with scenario 0
            # i.e., (0, 1), (0, 2), ..., (0, N) --> N - 1 constraints
            assert len(getattr(model, con_name)) == (len(model.scenarios) - 1)

        # Should not have any constraints sets beyond the length of design_vars - 1 (started with index 0)
        assert not hasattr(model, con_name_base + str(len(design_vars)))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_backward_design_fixing(self):
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
            assert hasattr(model, con_name)

            # Ensure that each set of constraints has all blocks pairs with scenario 0
            # i.e., (0, 1), (0, 2), ..., (0, N) --> N - 1 constraints
            assert len(getattr(model, con_name)) == (len(model.scenarios) - 1)

        # Should not have any constraints sets beyond the length of design_vars - 1 (started with index 0)
        assert not hasattr(model, con_name_base + str(len(design_vars)))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_forward_design_fixing(self):
        fd_method = "forward"
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
            assert hasattr(model, con_name)

            # Ensure that each set of constraints has all blocks pairs with scenario 0
            # i.e., (0, 1), (0, 2), ..., (0, N) --> N - 1 constraints
            assert len(getattr(model, con_name)) == (len(model.scenarios) - 1)

        # Should not have any constraints sets beyond the length of design_vars - 1 (started with index 0)
        assert not hasattr(model, con_name_base + str(len(design_vars)))

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_check_user_initialization(self):
        fd_method = "central"
        obj_used = "det"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_prior = np.ones((4, 4))
        FIM_initial = np.eye(4) + FIM_prior
        JAC_initial = np.ones((27, 4)) * 2
        L_initial = np.tril(
            np.ones((4, 4)) * 3
        )  # Must input lower triangular to get equality

        doe_obj = DesignOfExperiments(
            experiment,
            fd_formula=fd_method,
            step=1e-3,
            objective_option=obj_used,
            scale_constant_value=1,
            scale_nominal_param_value=True,
            prior_FIM=FIM_prior,
            jac_initial=JAC_initial,
            fim_initial=FIM_initial,
            L_initial=L_initial,
            L_LB=1e-7,
            solver=None,
            tee=False,
            args=None,
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        doe_obj.create_doe_model()

        # Grab the matrix values on the model
        FIM, FIM_prior_model, Q, L, sigma = get_FIM_FIMPrior_Q_L(doe_obj)

        # Make sure they match the inputs we gave
        assert np.array_equal(FIM, FIM_initial)
        assert np.array_equal(FIM_prior, FIM_prior_model)
        assert np.array_equal(L_initial, L)
        assert np.array_equal(JAC_initial, Q)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_update_FIM(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_update = np.ones((4, 4)) * 10

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

        doe_obj.create_doe_model()

        doe_obj.update_FIM_prior(FIM=FIM_update)

        # Grab values to ensure we set the correct piece
        FIM, FIM_prior_model, Q, L, sigma = get_FIM_FIMPrior_Q_L(doe_obj)

        # Make sure they match the inputs we gave
        assert np.array_equal(FIM_update, FIM_prior_model)

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_get_experiment_inputs_without_blocks(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_update = np.ones((4, 4)) * 10

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

        stuff = doe_obj.get_experiment_input_values(model=doe_obj.compute_FIM_model)

        assert len(stuff) == len(
            [k.name for k, v in doe_obj.compute_FIM_model.experiment_inputs.items()]
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_get_experiment_outputs_without_blocks(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_update = np.ones((4, 4)) * 10

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

        stuff = doe_obj.get_experiment_output_values(model=doe_obj.compute_FIM_model)

        assert len(stuff) == len(
            [k.name for k, v in doe_obj.compute_FIM_model.experiment_outputs.items()]
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_get_measurement_error_without_blocks(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_update = np.ones((4, 4)) * 10

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

        stuff = doe_obj.get_measurement_error_values(model=doe_obj.compute_FIM_model)

        assert len(stuff) == len(
            [k.name for k, v in doe_obj.compute_FIM_model.measurement_error.items()]
        )

    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_get_unknown_parameters_without_blocks(self):
        fd_method = "forward"
        obj_used = "trace"

        experiment = FullReactorExperiment(data_ex, 10, 3)

        FIM_update = np.ones((4, 4)) * 10

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

        # Make sure the values can be retrieved
        stuff = doe_obj.get_unknown_parameter_values(model=doe_obj.compute_FIM_model)

        assert len(stuff) == len(
            [k.name for k, v in doe_obj.compute_FIM_model.unknown_parameters.items()]
        )


if __name__ == "__main__":
    unittest.main()
