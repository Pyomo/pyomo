#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
from io import StringIO
import sys

import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.trustregion.examples import example1, example2
from pyomo.environ import SolverFactory

logger = logging.getLogger('pyomo.contrib.trustregion')


@unittest.skipIf(
    not SolverFactory('ipopt').available(False), "The IPOPT solver is not available"
)
class TestTrustRegionMethod(unittest.TestCase):
    def test_example1(self):
        # Check the log contents
        log_OUTPUT = StringIO()
        # Check the printed contents
        print_OUTPUT = StringIO()
        sys.stdout = print_OUTPUT
        with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
            example1.main()
        sys.stdout = sys.__stdout__
        # Check number of iterations - which should be 4 total
        self.assertIn('Iteration 0', log_OUTPUT.getvalue())
        self.assertIn('Iteration 4', log_OUTPUT.getvalue())
        self.assertNotIn('Iteration 5', log_OUTPUT.getvalue())
        # There were only theta-type steps in this
        self.assertIn('theta-type step', log_OUTPUT.getvalue())
        self.assertNotIn('f-type step', log_OUTPUT.getvalue())
        # These two pieces of information are only printed, not logged
        self.assertNotIn('EXIT: Optimal solution found.', log_OUTPUT.getvalue())
        self.assertNotIn('None :   True : 0.2770447887637415', log_OUTPUT.getvalue())
        # All of this should be printed
        self.assertIn('Iteration 0', print_OUTPUT.getvalue())
        self.assertIn('Iteration 4', print_OUTPUT.getvalue())
        self.assertNotIn('Iteration 5', print_OUTPUT.getvalue())
        self.assertIn('theta-type step', print_OUTPUT.getvalue())
        self.assertNotIn('f-type step', print_OUTPUT.getvalue())
        self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
        self.assertIn('None :   True : 0.2770447887637415', print_OUTPUT.getvalue())

    def test_example2(self):
        # Check the log contents
        log_OUTPUT = StringIO()
        # Check the printed contents
        print_OUTPUT = StringIO()
        sys.stdout = print_OUTPUT
        with LoggingIntercept(log_OUTPUT, 'pyomo.contrib.trustregion', logging.INFO):
            example2.main()
        sys.stdout = sys.__stdout__
        # Check the number of iterations - which should be 70ish, but not 80
        self.assertIn('Iteration 0', log_OUTPUT.getvalue())
        self.assertIn('Iteration 70', log_OUTPUT.getvalue())
        self.assertNotIn('Iteration 85', log_OUTPUT.getvalue())
        # This had all three step-types, so all three should be present
        self.assertIn('theta-type step', log_OUTPUT.getvalue())
        self.assertIn('f-type step', log_OUTPUT.getvalue())
        self.assertIn('step rejected', log_OUTPUT.getvalue())
        # These two pieces of information are only printed, not logged
        self.assertNotIn('EXIT: Optimal solution found.', log_OUTPUT.getvalue())
        self.assertNotIn('None :   True : 48.383116936949', log_OUTPUT.getvalue())
        # All of this should be printed
        self.assertIn('Iteration 0', print_OUTPUT.getvalue())
        self.assertIn('Iteration 70', print_OUTPUT.getvalue())
        self.assertNotIn('Iteration 85', print_OUTPUT.getvalue())
        self.assertIn('theta-type step', print_OUTPUT.getvalue())
        self.assertIn('f-type step', print_OUTPUT.getvalue())
        self.assertIn('step rejected', print_OUTPUT.getvalue())
        self.assertIn('EXIT: Optimal solution found.', print_OUTPUT.getvalue())
        self.assertIn('None :   True : 48.383116936949', print_OUTPUT.getvalue())
