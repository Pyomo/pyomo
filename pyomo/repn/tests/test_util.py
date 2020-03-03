import logging

import pyutilib.th as unittest
from six import StringIO

from pyomo.common.log import LoggingIntercept
from pyomo.repn.util import ftoa


class TestRepnUtils(unittest.TestCase):
    def test_ftoa(self):
        warning_output = StringIO()
        with LoggingIntercept(warning_output, 'pyomo.core', logging.WARNING):
            x1 = 1.123456789012345678
            x1str = ftoa(x1)
            self.assertEqual(x1str, '1.1234567890123457')
            # self.assertIn("to string resulted in loss of precision", warning_output.getvalue())
            # Not sure how to construct a case that hits that part of the code, but it should be done.
        x2 = 1.0 + 1E-15
        x2str = ftoa(x2)
        self.assertEqual(x2str, str(x2))


if __name__ == "__main__":
    unittest.main()
