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
                'import pyomo.core, sys; '
                'sys.exit( 1 if "pyomo.environ" in sys.modules else 0 )'])
        if rc:
            self.fail("Importing pyomo.core automatically imports "
                      "pyomo.environ and it should not.")

if __name__ == "__main__":
    unittest.main()

