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
    scipy,
    scipy_available,
    matplotlib,
    matplotlib_available,
)

import platform

is_osx = platform.mac_ver()[0] != ''

import pyomo.common.unittest as unittest
import os

import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest as parmestbase
import pyomo.environ as pyo

from pyomo.opt import SolverFactory

ipopt_available = SolverFactory('ipopt').available()

from pyomo.common.fileutils import find_library

pynumero_ASL_available = False if find_library('pynumero_ASL') is None else True

testdir = os.path.dirname(os.path.abspath(__file__))


@unittest.skipIf(
    not parmest.parmest_available,
    "Cannot test parmest: required dependencies are missing",
)
@unittest.skipIf(not ipopt_available, "The 'ipopt' command is not available")
class TestSolver(unittest.TestCase):
    def setUp(self):
        pass

    def test_ipopt_solve_with_stats(self):
        from pyomo.contrib.parmest.examples.rooney_biegler.rooney_biegler import (
            rooney_biegler_model,
        )
        from pyomo.contrib.parmest.utils import ipopt_solve_with_stats

        data = pd.DataFrame(
            data=[[1, 8.3], [2, 10.3], [3, 19.0], [4, 16.0], [5, 15.6], [7, 19.8]],
            columns=['hour', 'y'],
        )

        model = rooney_biegler_model(data)
        solver = pyo.SolverFactory('ipopt')
        solver.solve(model)

        status_obj, solved, iters, time, regu = ipopt_solve_with_stats(model, solver)

        self.assertEqual(solved, True)


if __name__ == '__main__':
    unittest.main()
