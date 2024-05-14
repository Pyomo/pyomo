#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

from pyomo.common.dependencies import attempt_import
from pyomo.contrib.cspline_external.cspline_parameters import (
    cubic_parameters_model,
    get_parameters,
)
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
import pyomo.common.unittest as unittest

np, numpy_available = attempt_import("numpy")

# REMOVE, temporary just to set solver path
import idaes


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

        params = os.path.join(this_file_dir(), "test.txt")
        knots, a0, a1, a2, a3 = get_parameters(m, params)
        assert len(knots) == 5
        assert len(a0) == 4
        assert len(a1) == 4
        assert len(a2) == 4
        assert len(a3) == 4

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
        # Check the value calculated by from parameters
        for i, x in enumerate(x_data):
            seg = i - 1
            if seg < 0:
                seg = 0
            y = a[0][seg] + a[1][seg] * x + a[2][seg] * x**2 + a[3][seg] * x**3
            self.assertAlmostEqual(y, y_data[i], 8)

        os.remove(params)
