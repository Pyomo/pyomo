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

import io
import os

from pyomo.contrib.cspline_external.cspline_parameters import (
    cubic_parameters_model,
    CsplineParameters,
)
from pyomo.opt import check_available_solvers
from pyomo.common.dependencies import numpy as np, numpy_available
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest


@unittest.skipUnless(
    check_available_solvers("ipopt"), "The 'ipopt' solver is not available"
)
@unittest.skipIf(not numpy_available, "numpy is not available.")
class CsplineExternalParamsTest(unittest.TestCase):
    def test_param_gen(self):
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 3, 5, 2, 1]

        m = cubic_parameters_model(x_data, y_data)

        # This is a linear system of equations by default, so although ipopt
        # is not needed, it works perfectly well.
        solver_obj = pyo.SolverFactory("ipopt")
        solver_obj.solve(m)

        params = CsplineParameters(model=m)

        # Make sure we have the expected number of parameters
        assert params.n_knots == 5
        assert params.n_segments == 4
        assert params.valid

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data))
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 8)

        # Make sure reading and writing parameters
        # to/from files works right.

        fptr = io.StringIO("")
        params.write_parameters(fptr)
        # Go to start of string stream
        fptr.seek(0, os.SEEK_SET)

        params.knots = np.array([1, 3])
        assert not params.valid
        params.get_parameters_from_file(fptr)
        assert params.valid

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data))
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 8)
