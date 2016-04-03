#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Test the Pyomo NL writer against the AMPL NL writer
#

import re
import glob
import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.subprocess

import pyomo.scripting.pyomo_main as main

# https://github.com/ghackebeil/gjh_asl_json
has_gjh_asl_json = False
if os.system('gjh_asl_json -v') == 0:
    has_gjh_asl_json = True

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
def nlwriter_baseline_test(self, name):
    if os.path.exists(currdir+name+'.dat'):
        self.pyomo(['--output='+currdir+name+'.test.nl',
                    currdir+name+'_testCase.py',
                    currdir+name+'.dat'])
    else:
        self.pyomo(['--output='+currdir+name+'.test.nl',
                    currdir+name+'_testCase.py'])

    # Check that the pyomo nl file matches its own baseline
    self.assertFileEqualsBaseline(currdir+name+'.test.nl', currdir+name+'.pyomo.nl', tolerance=1e-7)


class ASLTests(Tests):

    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)
ASLTests = unittest.category('smoke','nightly','expensive')(ASLTests)

#
# The following test calls the gjh_asl_json executable to
# generate JSON files corresponding to both the
# AMPL-generated nl file and the Pyomo-generated nl
# file. The JSON files are then diffed using the pyutilib.th
# test class method assertMatchesJsonBaseline()
#
@unittest.nottest
def nlwriter_asl_test(self, name):
    if not has_gjh_asl_json:
        self.skipTest("'gjh_asl_json' executable not available")
        return
    if os.path.exists(currdir+name+'.dat'):
        self.pyomo(['--output='+currdir+name+'.test.nl',
                    '--file-determinism=3',
                    '--symbolic-solver-labels',
                    currdir+name+'_testCase.py',
                    currdir+name+'.dat'])
    else:
        self.pyomo(['--output='+currdir+name+'.test.nl',
                    '--file-determinism=3',
                    '--symbolic-solver-labels',
                    currdir+name+'_testCase.py'])

    # compare AMPL and Pyomo nl file structure
    try:
        os.remove(currdir+name+'.ampl.json')
    except Exception:
        pass
    try:
        os.remove(currdir+name+'.test.json')
    except Exception:
        pass

    # obtain the nl file summary information for comparison with ampl
    p = pyutilib.subprocess.run(
        'gjh_asl_json '+currdir+name+'.test.nl rows='
        +currdir+name+'.test.row cols='+currdir+name+'.test.col')
    self.assertTrue(p[0] == 0, msg=p[1])

    # obtain the nl file summary information for comparison with pyomo
    p = pyutilib.subprocess.run(
        'gjh_asl_json '+currdir+name+'.ampl.nl rows='
        +currdir+name+'.ampl.row cols='+currdir+name+'.ampl.col')
    self.assertTrue(p[0] == 0, msg=p[1])

    self.assertMatchesJsonBaseline(
        currdir+name+'.test.json',
        currdir+name+'.ampl.json',
        tolerance=1e-8)

    os.remove(currdir+name+'.ampl.json')

    # delete temporary test files
    os.remove(currdir+name+'.test.col')
    os.remove(currdir+name+'.test.row')
    os.remove(currdir+name+'.test.nl')

# add test methods to classes
for f in glob.glob(currdir+'*_testCase.py'):
    name = re.split('[._]',os.path.basename(f))[0]
    BaselineTests.add_fn_test(fn=nlwriter_baseline_test, name=name)
    ASLTests.add_fn_test(fn=nlwriter_asl_test, name=name)

if __name__ == "__main__":
    unittest.main()
