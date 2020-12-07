#
# Only run the tests in this package if the pyomo.contrib.example package
# has been successfully imported.
#
import os
import sys
import pyomo.contrib.example
import pyutilib.th as unittest


class Tests(unittest.TestCase):

    def test1(self):
        pass


if __name__ == "__main__":
    unittest.main()
