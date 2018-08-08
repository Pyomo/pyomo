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
import pyutilib.subprocess

import pyomo.scripting.pyomo_main as main


class Tests(unittest.TestCase):

    def pyomo(self, cmd):
        os.chdir(currdir)
        output = main.main(['convert', '--logging=quiet', '-c']+cmd)
        return output

class BaselineTests(Tests):
    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)
BaselineTests = unittest.category('smoke', 'nightly','expensive')(BaselineTests)

#
#The following test generates an nl file for the test case
#and checks that it matches the current pyomo baseline nl file
#
@unittest.nottest
def gams_writer_baseline_test(self, name):
    if os.path.exists(datadir+name+'.dat'):
        self.pyomo(['--output='+currdir+name+'.test.gms',
                    datadir+name+'_testCase.py',
                    datadir+name+'.dat'])
    else:
        self.pyomo(['--output='+currdir+name+'.test.gms',
                    datadir+name+'_testCase.py'])

    # Check that the pyomo nl file matches its own baseline
    self.assertFileEqualsBaseline(
        currdir+name+'.test.gms', currdir+name+'.pyomo.gms',
        tolerance=(1e-7, False))


class ASLTests(Tests):

    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)
ASLTests = unittest.category('smoke','nightly','expensive')(ASLTests)


# add test methods to classes
for f in glob.glob(datadir+'*_testCase.py'):
    name = re.split('[._]',os.path.basename(f))[0]
    BaselineTests.add_fn_test(fn=gams_writer_baseline_test, name=name)
    #ASLTests.add_fn_test(fn=nlwriter_asl_test, name=name)

if __name__ == "__main__":
    unittest.main()
