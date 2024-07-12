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

f = open('result.json')
data_ex = json.load(f)
data_ex['control_points'] = {float(k): v for k, v in data_ex['control_points'].items()}

class TestReactorExampleModel(unittest.TestCase):
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