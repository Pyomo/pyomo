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
    numpy as np,
    numpy_available,
    pandas as pd,
    pandas_available,
)

import pyomo.common.unittest as unittest
from pyomo.contrib.doe import DesignOfExperiments
from pyomo.contrib.doe.tests import doe_test_example
from pyomo.contrib.doe.doe import _SMALL_TOLERANCE_IMG, compute_FIM_metrics
from pyomo.opt import SolverFactory
import pyomo.environ as pyo

import json
from pathlib import Path
import idaes.core.solvers.get_solver

ipopt_available = SolverFactory("ipopt").available()
k_aug_available = SolverFactory("k_aug", solver_io="nl", validate=False)


@unittest.skipIf(not numpy_available, "Numpy is not available")
class TestFullFactorialMetrics(unittest.TestCase):
    def test_compute_FIM_full_factorial_metrics(self):
        # Create a sample Fisher Information Matrix (FIM)
        FIM = np.array([[4, 2], [2, 3]])


# ======================================================================
import numpy as np


def test_FIM_metrics():
    # Create a sample Fisher Information Matrix (FIM)
    FIM = np.array([[4, 2], [2, 3]])

    # Call the function to compute metrics
    results = compute_FIM_metrics(FIM)

    # Use known values for assertions
    det_expected = np.linalg.det(FIM)
    D_opt_expected = np.log10(det_expected)

    trace_expected = np.trace(FIM)
    A_opt_expected = np.log10(trace_expected)

    E_vals_expected, _ = np.linalg.eig(FIM)
    min_eigval = np.min(E_vals_expected.real)

    cond_expected = np.linalg.cond(FIM)

    assert np.isclose(results["det_FIM"], det_expected)
    assert np.isclose(results["trace_FIM"], trace_expected)
    assert np.allclose(results["E_vals"], E_vals_expected)
    assert np.isclose(results["D_opt"], D_opt_expected)
    assert np.isclose(results["A_opt"], A_opt_expected)
    if min_eigval.real > 0:
        assert np.isclose(results["E_opt"], np.log10(min_eigval))
    else:
        assert np.isnan(results["E_opt"])

    assert np.isclose(results["ME_opt"], np.log10(cond_expected))


def test_FIM_metrics_warning_printed(capfd):
    # Create a matrix with an imaginary component large enough to trigger the warning
    FIM = np.array([[9, -2], [9, 3]])

    # Call the function
    compute_FIM_metrics(FIM)

    # Capture stdout and stderr
    out, err = capfd.readouterr()

    # Correct expected message
    expected_message = "Eigenvalue has imaginary component greater than {}, contact developers if this issue persists."

    # Ensure expected message is in the output
    assert expected_message in out
