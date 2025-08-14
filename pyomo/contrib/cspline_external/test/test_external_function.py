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
import os
import platform
import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.common.fileutils import find_library, this_file_dir

_lib = find_library("cspline_external")
is_pypy = platform.python_implementation().lower().startswith("pypy")


@unittest.skipIf(is_pypy, "Cannot evaluate external functions under pypy")
@unittest.skipIf(not _lib, "cspline library is not available.")
class CsplineExternal1DTest(unittest.TestCase):

    def test_function_call(self):
        """Test that the cspline behaves as expected"""
        # The parameters for the function call are stored in
        # this file. The file is only read once by the external
        # function on the first call.
        params = os.path.join(this_file_dir(), "test_params.txt")
        # this is that data that was used to generate the params
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 3, 5, 2, 1]
        # Model with the cspline function
        m = pyo.ConcreteModel()
        m.f = pyo.ExternalFunction(library=_lib, function="cspline")
        m.x = pyo.Var(initialize=2)
        m.e = pyo.Expression(expr=m.f(m.x, params))
        # We know that spline should go through the data points
        # and that f(x) f'(x) and f''(x) for the right and left
        # segment at a knot should be the same.
        for x, y in zip(x_data, y_data):
            delta = 1e-5
            f, fx, fxx = m.f.evaluate_fgh(args=(x, params))
            f_left, fx_left, fxx_left = m.f.evaluate_fgh(args=(x - delta, params))
            f_right, fx_right, fxx_right = m.f.evaluate_fgh(args=(x + delta, params))
            self.assertAlmostEqual(f, y, 8)
            # check left/right
            self.assertAlmostEqual(f_left, y, 3)
            self.assertAlmostEqual(f_right, y, 3)
            # check left/right derivatives
            self.assertAlmostEqual(fx_left[0], fx[0], 3)
            self.assertAlmostEqual(fx_right[0], fx[0], 3)
            self.assertAlmostEqual(fxx_left[0], fxx[0], 3)
            self.assertAlmostEqual(fxx_right[0], fxx[0], 3)
        # we know the endpoint constraints are that f''(0) = 0
        f, fx, fxx = m.f.evaluate_fgh(args=(x_data[0], params))
        self.assertAlmostEqual(fxx[0], 0, 8)
        f, fx, fxx = m.f.evaluate_fgh(args=(x_data[-1], params))
        self.assertAlmostEqual(fxx[0], 0, 8)
        # check a little more in model context
        self.assertAlmostEqual(pyo.value(m.e), 3, 8)

    def test_function_call_form2(self):
        """Test that the cspline can take the full parameter file contents in str"""
        params = os.path.join(this_file_dir(), "test_params.txt")
        with open(params, "r") as fptr:
            params = fptr.read()
        # this is that data that was used to generate the params
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 3, 5, 2, 1]
        # Model with the cspline function
        m = pyo.ConcreteModel()
        m.f = pyo.ExternalFunction(library=_lib, function="cspline")
        m.x = pyo.Var(initialize=2)
        m.e = pyo.Expression(expr=m.f(m.x, params))
        # We know that spline should go through the data points
        # and that f(x) f'(x) and f''(x) for the right and left
        # segment at a knot should be the same.
        for x, y in zip(x_data, y_data):
            delta = 1e-5
            f, fx, fxx = m.f.evaluate_fgh(args=(x, params))
            f_left, fx_left, fxx_left = m.f.evaluate_fgh(args=(x - delta, params))
            f_right, fx_right, fxx_right = m.f.evaluate_fgh(args=(x + delta, params))
            self.assertAlmostEqual(f, y, 8)
            # check left/right
            self.assertAlmostEqual(f_left, y, 3)
            self.assertAlmostEqual(f_right, y, 3)
            # check left/right derivatives
            self.assertAlmostEqual(fx_left[0], fx[0], 3)
            self.assertAlmostEqual(fx_right[0], fx[0], 3)
            self.assertAlmostEqual(fxx_left[0], fxx[0], 3)
            self.assertAlmostEqual(fxx_right[0], fxx[0], 3)
        # we know the endpoint constraints are that f''(0) = 0
        f, fx, fxx = m.f.evaluate_fgh(args=(x_data[0], params))
        self.assertAlmostEqual(fxx[0], 0, 8)
        f, fx, fxx = m.f.evaluate_fgh(args=(x_data[-1], params))
        self.assertAlmostEqual(fxx[0], 0, 8)
        # check a little more in model context
        self.assertAlmostEqual(pyo.value(m.e), 3, 8)

    def test_ampl_derivatives(self):
        # Make sure the function values and derivatives are right
        # based on parameters. This time look at segment mid-points

        # The parameters for the function call are stored in
        # this file. The file is only read once by the external
        # function on the first call.
        params = os.path.join(this_file_dir(), "test_params.txt")
        # this is that data that was used to generate the params
        x_data = [1, 2, 3, 4, 5]
        y_data = [2, 3, 5, 2, 1]
        # Model with the cspline function
        m = pyo.ConcreteModel()
        m.f = pyo.ExternalFunction(library=_lib, function="cspline")
        # Read parameters
        with open(params, "r") as fptr:
            # line 1: number of knots
            ns = int(fptr.readline())
            # Make param lists
            knots = [None] * (ns + 1)
            a = [None] * 4
            for j in range(4):
                a[j] = [None] * ns

            # Read params
            for i in range(ns + 1):
                knots[i] = float(fptr.readline())
            for j in range(4):
                for i in range(ns):
                    a[j][i] = float(fptr.readline())
        # Check the value calculated by from parameters to external
        # function for each segment.  Use the mid.
        for i in range(ns):
            x = (x_data[i] + x_data[i + 1]) / 2.0
            y = a[0][i] + a[1][i] * x + a[2][i] * x**2 + a[3][i] * x**3
            yx = a[1][i] + 2 * a[2][i] * x + 3 * a[3][i] * x**2
            yxx = 2 * a[2][i] + 6 * a[3][i] * x
            f, fx, fxx = m.f.evaluate_fgh(args=(x, params))
            self.assertAlmostEqual(f, y, 8)
            self.assertAlmostEqual(fx[0], yx, 8)
            self.assertAlmostEqual(fxx[0], yxx, 8)

    def test_load_multiple_splines(self):
        # The last test ensures that you can load multiple splines
        # for a model without trouble.  You should be able to use
        # as many parameter set as you want.  The external function
        # only reads a file once as long as the library remains loaded.

        # first spline is the one we used before
        params1 = os.path.join(this_file_dir(), "test_params.txt")
        # second spline is a line with intercept = 0, slope = 1
        params2 = os.path.join(this_file_dir(), "test_params_line.txt")
        # Model with the cspline function
        m = pyo.ConcreteModel()
        m.f = pyo.ExternalFunction(library=_lib, function="cspline")
        m.x = pyo.Var(initialize=1)
        m.e1 = pyo.Expression(expr=m.f(m.x, params1))
        m.e2 = pyo.Expression(expr=m.f(m.x, params2))

        # Make sure both functions return correct value
        self.assertAlmostEqual(pyo.value(m.e1), 2, 8)
        self.assertAlmostEqual(pyo.value(m.e2), 1, 8)

        # make sure loading the second set of parameters
        # didn't mess up the first
        self.assertAlmostEqual(pyo.value(m.e1), 2, 8)
        self.assertAlmostEqual(pyo.value(m.e2), 1, 8)
