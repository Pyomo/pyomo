#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
import sys

from importlib import import_module
from six import StringIO

from pyomo.common.log import LoggingIntercept

def force_load(module):
    if module in sys.modules:
        del sys.modules[module]
    return import_module(module)

class TestDeprecatedModules(unittest.TestCase):
    def test_rangeset(self):
        log = StringIO()
        with LoggingIntercept(log):
            from pyomo.core.base.set import RangeSet
        self.assertEqual(log.getvalue(), "")

        log = StringIO()
        with LoggingIntercept(log, 'pyomo'):
            rs = force_load('pyomo.core.base.rangeset')
        self.assertIn("The pyomo.core.base.rangeset module is deprecated.",
                      log.getvalue().strip().replace('\n',' '))
        self.assertIs(RangeSet, rs.RangeSet)

        # Run this twice to implicitly test the force_load() implementation
        log = StringIO()
        with LoggingIntercept(log, 'pyomo'):
            rs = force_load('pyomo.core.base.rangeset')
        self.assertIn("The pyomo.core.base.rangeset module is deprecated.",
                      log.getvalue().strip().replace('\n',' '))
        self.assertIs(RangeSet, rs.RangeSet)
        
