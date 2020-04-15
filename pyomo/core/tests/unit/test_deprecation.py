#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os
import pyutilib.th as unittest

from six import StringIO

from pyomo.common.log import LoggingIntercept

class TestDeprecatedModules(unittest.TestCase):
    def test_rangeset(self):
        log = StringIO()
        with LoggingIntercept(log):
            from pyomo.core.base.set import RangeSet
        self.assertEqual(log.getvalue(), "")
        with LoggingIntercept(log):
            from pyomo.core.base.rangeset import RangeSet as tmp_RS
        self.assertIn("The pyomo.core.base.rangeset module is deprecated.",
                      log.getvalue().strip().replace('\n',' '))
        self.assertIs(RangeSet, tmp_RS)
        
