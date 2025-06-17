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

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data) + 1e-4)
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 3)

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data) - 1e-4)
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 3)

        self.assertAlmostEqual(1, params.f(5), 8)
        self.assertAlmostEqual(1, params.f(5 - 1e-4), 3)
        self.assertAlmostEqual(1, params.f(5 + 1e-4), 3)
        self.assertAlmostEqual(2, params.f(1), 8)
        self.assertAlmostEqual(2, params.f(1 - 1e-4), 3)
        self.assertAlmostEqual(2, params.f(1 + 1e-4), 3)

    def test_param_increasing(self):
        x_data = [1, 2, 3, 4, 5]
        y_data = [1, 4, 9, 16, 25]

        m = cubic_parameters_model(x_data, y_data, objective_form=True)
        m.add_increasing_constraints()
        solver_obj = pyo.SolverFactory("ipopt")
        solver_obj.solve(m)

        params = CsplineParameters(model=m)

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data))
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 6)

        x_new = [1.5, 2.5, 3.5, 4.5]
        y_new = [2.25, 6.25, 12.25, 20.25]
        # Check midpoints.  These aren't necessarily super close due
        # to the end point second derivative constraints, and I
        # calculated test values from a quadratic.
        y_pred = params.f(np.array(x_new))
        for yd, yp in zip(y_new, y_pred):
            self.assertAlmostEqual(yd, yp, 0)

    def test_param_increasing_linear_extrap(self):
        x_data = [1, 2, 3, 4, 5]
        y_data = [1, 4, 9, 16, 25]

        m = cubic_parameters_model(x_data, y_data, objective_form=True)
        m.add_increasing_constraints()

        solver_obj = pyo.SolverFactory("ipopt")
        solver_obj.solve(m)

        params = CsplineParameters(model=m)
        params.add_linear_extrapolation_segments()

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data))
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 6)

        x_new = [0, 6, 7, 8]
        y_new = [-1.571, 34.42857, 43.8571, 53.2857]
        y_pred = params.f(np.array(x_new))
        for yd, yp in zip(y_new, y_pred):
            self.assertAlmostEqual(yd, yp, 3)

    def test_param_decreasing(self):
        x_data = [-1, -2, -3, -4, -5]
        y_data = [1, 4, 9, 16, 25]

        m = cubic_parameters_model(x_data, y_data, objective_form=True)
        m.add_decreasing_constraints()

        solver_obj = pyo.SolverFactory("ipopt")
        solver_obj.solve(m)

        params = CsplineParameters(model=m)

        # Make sure the predictions are correct
        y_pred = params.f(np.array(x_data))
        for yd, yp in zip(y_data, y_pred):
            self.assertAlmostEqual(yd, yp, 5)

        x_new = [-1.5, -2.5, -3.5, -4.5]
        y_new = [2.25, 6.25, 12.25, 20.25]
        # Check midpoints.  These aren't necessarily super close due
        # to the end point second derivative constraints, and I
        # calculated test values from a quadratic.
        y_pred = params.f(np.array(x_new))
        for yd, yp in zip(y_new, y_pred):
            self.assertAlmostEqual(yd, yp, 1)

    def test_convex(self):
        x_data = [-1, 1, 2, 3, 4, 5, 6]
        y_data = [1, 1, 4, 11, 16, 25, 36]

        m = cubic_parameters_model(
            x_data, y_data, objective_form=True, end_point_constraint=False
        )
        m.add_convex_constraints(tol=0.1)
        solver_obj = pyo.SolverFactory("ipopt")
        solver_obj.solve(m)
        params = CsplineParameters(model=m)
        y_pred = params.f(np.array(x_data))
        y_new = [1.007, 0.891, 4.524, 10.154, 16.536, 24.866, 36.0223]
        for yd, yp in zip(y_new, y_pred):
            self.assertAlmostEqual(yd, yp, 2)

    def test_concave(self):
        x_data = [1, 1, 2, 3, 4, 5, 6]
        y_data = [-1, -1, -4, -14, -16, -25, -36]
        m = cubic_parameters_model(
            x_data, y_data, objective_form=True, end_point_constraint=False
        )
        m.add_concave_constraints(tol=0.1)
        solver_obj = pyo.SolverFactory("ipopt")
        solver_obj.solve(m)
        params = CsplineParameters(model=m)
        y_pred = params.f(np.array(x_data))
        y_new = [-1.000, -1.000, -5.259, -11.3533, -17.5475, -24.80776, -36.032]
        for yd, yp in zip(y_new, y_pred):
            self.assertAlmostEqual(yd, yp, 2)
