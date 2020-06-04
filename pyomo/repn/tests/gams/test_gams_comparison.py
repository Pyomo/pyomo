#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test the Pyomo BAR writer
#

import re
import glob
import os
from os.path import abspath, dirname, join
currdir = dirname(abspath(__file__))+os.sep
datadir = abspath(join(currdir, "..", "ampl"))+os.sep

import pyutilib.th as unittest

import pyomo.scripting.pyomo_main as main


class Tests(unittest.TestCase):

    def pyomo(self, cmd):
        os.chdir(currdir)
        output = main.main(['convert', '--logging=quiet', '-c']+cmd)
        return output

class BaselineTests(Tests):
    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)

#
# The following test generates a GMS file for the test case
# and checks that it matches the current pyomo baseline GMS file
#
@unittest.nottest
def gams_writer_baseline_test(self, name, targetdir):
    if os.path.exists(targetdir+name+'.dat'):
        self.pyomo(['--output='+currdir+name+'.test.gms',
                    targetdir+name+'_testCase.py',
                    targetdir+name+'.dat'])
    else:
        self.pyomo(['--output='+currdir+name+'.test.gms',
                    targetdir+name+'_testCase.py'])

    # Check that the pyomo nl file matches its own baseline
    self.assertFileEqualsBaseline(
        currdir+name+'.test.gms', currdir+name+'.pyomo.gms',
        tolerance=(1e-7, False))

@unittest.nottest
def gams_writer_test_invalid(self, name, targetdir):
    with self.assertRaisesRegexp(
            RuntimeError, "GAMS files cannot represent the unary function"):
        if os.path.exists(targetdir+name+'.dat'):
            self.pyomo(['--output='+currdir+name+'.test.gms',
                        targetdir+name+'_testCase.py',
                        targetdir+name+'.dat'])
        else:
            self.pyomo(['--output='+currdir+name+'.test.gms',
                        targetdir+name+'_testCase.py'])

# add test methods to classes
invalid_tests = {'small14',}
for f in glob.glob(datadir+'*_testCase.py'):
    name = re.split('[._]',os.path.basename(f))[0]
    if name in invalid_tests:
        fcn = lambda s,name: gams_writer_test_invalid(s,name,datadir)
    else:
        fcn = lambda s,name: gams_writer_baseline_test(s,name,datadir)
    BaselineTests.add_fn_test(fn=fcn, name=name)

for f in glob.glob(currdir+'*_testCase.py'):
    name = re.split('[._]',os.path.basename(f))[0]
    fcn = lambda s,name: gams_writer_baseline_test(s,name,currdir)
    BaselineTests.add_fn_test(fn=fcn, name=name)

if __name__ == "__main__":
    unittest.main()
