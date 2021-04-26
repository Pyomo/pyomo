#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging

import pyomo.common.unittest as unittest
from io import StringIO

from pyomo.common.log import LoggingIntercept
from pyomo.repn.util import ftoa

try:
    import numpy as np
    numpy_available = True
except:
    numpy_available = False

class TestRepnUtils(unittest.TestCase):
    def test_ftoa(self):
        # Test that trailing zeros are removed
        f = 1.0
        a = ftoa(f)
        self.assertEqual(a, '1')

    @unittest.skipIf(not numpy_available, "NumPy is not available")
    def test_ftoa_precision(self):
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.core', logging.WARNING):
            f = np.longdouble('1.1234567890123456789')
            a = ftoa(f)
        self.assertEqual(a, '1.1234567890123457')
        # Depending on the platform, np.longdouble may or may not have
        # higher precision than float:
        if f == float(f):
            test = self.assertNotRegexpMatches
        else:
            test = self.assertRegex
        test( log.getvalue(),
              '.*Converting 1.1234567890123456789 to string '
              'resulted in loss of precision' )

if __name__ == "__main__":
    unittest.main()
