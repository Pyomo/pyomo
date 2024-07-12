from pyomo.common.dependencies import (
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)

from pyomo.contrib.doe.tests.experiment_class_example_flags import *
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

class TestReactorExampleErrors(unittest.TestCase):
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_check_no_experiment_outputs(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 1  # Value for faulty model build mode - 1: No exp outputs

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
            args={'flag': flag_val},
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        with self.assertRaisesRegex(
            RuntimeError, "Experiment model does not have suffix " + '"experiment_outputs".'
        ):
            doe_obj.create_doe_model()
    
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_check_no_measurement_error(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 2  # Value for faulty model build mode - 2: No meas error

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
            args={'flag': flag_val},
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        with self.assertRaisesRegex(
            RuntimeError, "Experiment model does not have suffix " + '"measurement_error".'
        ):
            doe_obj.create_doe_model()
    
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_check_no_experiment_inputs(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 3  # Value for faulty model build mode - 3: No exp inputs/design vars

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
            args={'flag': flag_val},
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        with self.assertRaisesRegex(
            RuntimeError, "Experiment model does not have suffix " + '"experiment_inputs".'
        ):
            doe_obj.create_doe_model()
    
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_check_no_unknown_parameters(self):
        fd_method = "central"
        obj_used = "trace"
        flag_val = 4  # Value for faulty model build mode - 4: No unknown params

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
            args={'flag': flag_val},
            _Cholesky_option=True,
            _only_compute_fim_lower=True,
        )

        with self.assertRaisesRegex(
            RuntimeError, "Experiment model does not have suffix " + '"unknown_parameters".'
        ):
            doe_obj.create_doe_model()

if __name__ == "__main__":
    unittest.main()
