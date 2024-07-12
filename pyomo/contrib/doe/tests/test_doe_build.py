from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)

from experiment_class_example import *
from pyomo.contrib.doe import *


import pyomo.common.unittest as unittest

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory("ipopt").available()

f = open("result.json")
data_ex = json.load(f)
data_ex["control_points"] = {float(k): v for k, v in data_ex["control_points"].items()}


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


if __name__ == "__main__":
    unittest.main()
