import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep
datadir = currdir+'data'+os.sep+'osrlFiles'+os.sep

import glob
from nose.tools import nottest
import pyutilib.th as unittest
import pyutilib.services
import pyomo.opt
import pyomo.modeling

old_tempdir = pyutilib.services.TempfileManager.tempdir

class Test(unittest.TestCase):

    def setUp(self):
        pyutilib.services.TempfileManager.tempdir = currdir

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()
        pyutilib.services.TempfileManager.tempdir = old_tempdir


@nottest
def test_osrl(self, name):
    reader = pyomo.opt.ReaderFactory("osrl")
    soln = reader(datadir+name+'.osrl')
    soln.write(filename=datadir+name+'_out.json', format='json')
    self.assertMatchesJsonBaseline(datadir+name+"_out.json", datadir+name+"_baseline.json")

for fname in glob.glob(currdir+'data/osrlFiles/*.osrl'):
    name = os.path.basename(fname)
    name = name.split('.')[0]
    Test.add_fn_test(fn=test_osrl, name=name)

if __name__ == "__main__":
    unittest.main()
