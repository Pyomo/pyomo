#
# Only run the tests in this package if the TEST_PYOMO_CONTRIB environment
# variable has been set to '1'.  Otherwise, skip the tests.
#
import os
import sys
if os.environ.get('TEST_PYOMO_CONTRIB', 0):
    import pyomo.contrib.example
    run = True
else:
    run = False

import pyutilib.th as unittest


@unittest.skipIf(run == False, "The TEST_PYOMO_CONTRIB environment is not set.")
class Tests(unittest.TestCase):

    def test1(self):
        env = os.environ.get('TEST_PYOMO_CONTRIB', 0)
        self.assertNotEqual(env, 0)


if __name__ == "__main__":
    unittest.main()
