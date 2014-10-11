#
# Unit Tests for pyomo.base.misc
#
import sys
import pyutilib.th as unittest
from pyutilib.subprocess import run_command

class TestPyomoEnviron(unittest.TestCase):
    def test_not_auto_imported(self):
        rc, output = run_command([
                sys.executable, '-c', 
                'import pyomo.pyomo, sys; '
                'sys.exit( 1 if "pyomo.modeling" in sys.modules else 0 )'])
        if rc:
            self.fail("Importing pyomo.core automatically imports "
                      "pyomo.modeling and it should not.")
