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

from pyomo.common.dependencies import (
    pathlib,
    numpy as np,
    numpy_available,
)
from pyomo.contrib.doe.examples.reactor_experiment import ReactorExperiment
from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo
import idaes.core.solvers.get_solver

import pyomo.common.unittest as unittest

import json


# Example for sensitivity analysis on the reactor experiment
# After sensitivity analysis is done, we perform optimal DoE
def run_reactor_doe(test_compute_FIM_full_factorial=True):
    # Read in file
    DATA_DIR = pathlib.Path(__file__).parents[1]
    file_path = DATA_DIR / "examples/result.json"

    with open(file_path) as f:
        data_ex = json.load(f)

    # Put temperature control time points into correct format for reactor experiment
    data_ex["control_points"] = {
        float(k): v for k, v in data_ex["control_points"].items()
    }

    # Create a ReactorExperiment object; data and discretization information are part
    # of the constructor of this object
    experiment = ReactorExperiment(data=data_ex, nfe=10, ncp=3)

    # Use a central difference, with step size 1e-3
    fd_formula = "central"
    step_size = 1e-3

    # Use the determinant objective with scaled sensitivity matrix
    objective_option = "determinant"
    scale_nominal_param_value = True

    # Create the DesignOfExperiments object
    doe_obj = DesignOfExperiments(
        experiment,
        fd_formula=fd_formula,
        step=step_size,
        objective_option=objective_option,
        scale_constant_value=1,
        scale_nominal_param_value=scale_nominal_param_value,
        prior_FIM=None,
        jac_initial=None,
        fim_initial=None,
        L_diagonal_lower_bound=1e-7,
        solver=None,
        tee=False,
        get_labeled_model_args=None,
        _Cholesky_option=True,
        _only_compute_fim_lower=True,
    )

    if test_compute_FIM_full_factorial:
        # Make design ranges to compute the full factorial design
        design_ranges = {"CA[0]": [1, 5, 2], "T[0]": [300, 700, 2]}

        # Compute the full factorial design with the sequential FIM calculation
        ff_results = doe_obj.compute_FIM_full_factorial(
            design_ranges=design_ranges, method="sequential"
        )
        return ff_results


@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestDoe(unittest.TestCase):
    def test_doe_full_factorial(self):
        log10_D_opt_expected = [
            np.float64(-13.321347741255337),
            np.float64(3.8035612211158707),
            np.float64(-7.724323094449262),
            np.float64(9.395321258173526),
        ]

        log10_A_opt_expected = [
            np.float64(3.5646581425454578),
            np.float64(2.922649226588521),
            np.float64(4.962598150652743),
            np.float64(4.3205892352904876),
        ]

        log10_E_opt_expected = [
            np.float64(-10.076931572437823),
            np.float64(-0.6660428224151175),
            np.float64(-8.67332037872937),
            np.float64(0.731897189777441),
        ]
        log10_ME_opt_expected = [
            np.float64(13.51143310646149),
            np.float64(3.570243133023128),
            np.float64(13.505430874322686),
            np.float64(3.5702431295446915),
        ]

        eigval_min_expected = [
            np.float64(8.376612538754303e-11),
            np.float64(0.21575316611777548),
            np.float64(2.1216787236688646e-09),
            np.float64(5.393829196668378),
        ]

        eigval_max_expected = [
            np.float64(2714.297914184112),
            np.float64(802.0479084262055),
            np.float64(67857.4478581609),
            np.float64(20051.197712596462),
        ]

        det_FIM_expected = [
            np.float64(4.7714706717649e-14),
            np.float64(6361.524749138681),
            np.float64(1.886587295622232e-08),
            np.float64(2484970618.69026),
        ]

        trace_FIM_expected = [
            np.float64(3669.9330583293095),
            np.float64(836.8530948725596),
            np.float64(91748.32633892389),
            np.float64(20921.327373255765),
        ]
        ff_results = run_reactor_doe(test_compute_FIM_full_factorial=True)

        self.assertTrue(np.allclose(ff_results["log10 D-opt"], log10_D_opt_expected))
        self.assertTrue(np.allclose(ff_results["log10 A-opt"], log10_A_opt_expected))
        self.assertTrue(np.allclose(ff_results["log10 E-opt"], log10_E_opt_expected))
        self.assertTrue(np.allclose(ff_results["log10 ME-opt"], log10_ME_opt_expected))
        self.assertTrue(np.allclose(ff_results["eigval_min"], eigval_min_expected))
        self.assertTrue(np.allclose(ff_results["eigval_max"], eigval_max_expected))
        self.assertTrue(np.allclose(ff_results["det_FIM"], det_FIM_expected))
        self.assertTrue(np.allclose(ff_results["trace_FIM"], trace_FIM_expected))


if __name__ == "__main__":
    unittest.main()
