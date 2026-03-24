# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import pyomo.environ as pyo
import pyomo.devel.initialization as ini
from pyomo.devel.initialization.examples.init_polynomial_ex import main
from pyomo.common import unittest
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import SolutionStatus


scip = SolverFactory('scip_direct')
ipopt = SolverFactory('ipopt')


@unittest.skipUnless(scip.available(), 'scip is not available')
@unittest.skipUnless(ipopt.available(), 'ipopt is not available')
class TestExamples(unittest.TestCase):
    def test_poly_global(self):
        stat, x = main(method=ini.InitializationMethod.global_opt)
        self.assertEqual(stat, SolutionStatus.optimal)
        self.assertAlmostEqual(x, -9.920159607881597)

    def test_poly_pwl(self):
        stat, x = main(method=ini.InitializationMethod.pwl_approximation)
        self.assertEqual(stat, SolutionStatus.optimal)
        self.assertAlmostEqual(x, -9.920159607881597)

    def test_poly_lp(self):
        stat, x = main(method=ini.InitializationMethod.lp_approximation)
        self.assertEqual(stat, SolutionStatus.optimal)
        self.assertAlmostEqual(x, -9.920159607881597)
