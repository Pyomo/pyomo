#
# Unit Tests for coopr.opt.base.OS
#
#

import os
from os.path import abspath, dirname
cooprdir = dirname(abspath(__file__))+os.sep+".."+os.sep+".."+os.sep
currdir = dirname(abspath(__file__))+os.sep

from nose.tools import nottest
import xml
import pyutilib.th as unittest
import pyutilib.services
import coopr.opt
import coopr
import coopr.environ

old_tempdir = pyutilib.services.TempfileManager.tempdir

class Test(unittest.TestCase):

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir
        if os.path.exists(currdir+"test_sol.txt"):
            os.remove(currdir+"test_sol.txt")

    def test_factory(self):
        reader = coopr.opt.ReaderFactory("sol")
        if reader is None:
            raise IOError("Reader 'sol' is not registered")
        soln = reader(currdir+"test4_sol.sol", suffixes=["dual"])
        soln.write(filename=currdir+"factory.txt", format='json')
        self.assertMatchesJsonBaseline(currdir+"factory.txt", currdir+"test4_sol.jsn")


if __name__ == "__main__":
    unittest.main()
