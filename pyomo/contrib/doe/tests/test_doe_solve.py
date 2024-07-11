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

class TestReactorExamples(unittest.TestCase):
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
        
        assert (doe_obj.results['Solver Status'] == "ok")
    
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_reactor_fd_forward_solve(self):
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
        
        doe_obj.run_doe()
        
        assert (doe_obj.results['Solver Status'] == "ok")
    
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
        
        assert (doe_obj.results['Solver Status'] == "ok")
    
    # TODO: Fix determinant objective code, something is awry
    #       Should only be using Cholesky=True
    # @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    # @unittest.skipIf(not numpy_available, "Numpy is not available")
    # def test_reactor_obj_det_solve(self):
        # fd_method = "central"
        # obj_used = "det"
        
        # experiment = FullReactorExperiment(data_ex, 10, 3)
        
        # doe_obj = DesignOfExperiments(
            # experiment, 
            # fd_formula=fd_method,
            # step=1e-3,
            # objective_option=obj_used,
            # scale_constant_value=1,
            # scale_nominal_param_value=True,
            # prior_FIM=None,
            # jac_initial=None,
            # fim_initial=None,
            # L_initial=None,
            # L_LB=1e-7,
            # solver=None,
            # tee=False,
            # args=None,
            # _Cholesky_option=False,
            # _only_compute_fim_lower=False,
        # )
        
        # doe_obj.run_doe()
        
        # assert (doe_obj.results['Solver Status'] == "ok")
    
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
        
        assert (doe_obj.results['Solver Status'] == "ok")
    
    @unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
    @unittest.skipIf(not numpy_available, "Numpy is not available")
    def test_compute_FIM_seq(self):
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
        
        design_ranges = {
            'CA[0]': [1, 5, 3], 
            'T[0]': [300, 700, 3],
        }
        
        doe_obj.compute_FIM(method='sequential')

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
        
        design_ranges = {
            'CA[0]': [1, 5, 3], 
            'T[0]': [300, 700, 3],
        }
        
        doe_obj.compute_FIM_full_factorial(design_ranges=design_ranges, method='sequential')
        
    

if __name__ == "__main__":
    unittest.main()