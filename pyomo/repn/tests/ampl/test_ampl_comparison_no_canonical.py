#
# Test the Pyomo NL writer against the AMPL NL writer
#

import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import re
import glob


import pyutilib.th as unittest
import pyutilib.subprocess
import pyomo.scripting.pyomo as main
import json

has_asl_test = False
if os.system('asl_test -v') == 0:
    has_asl_test = True
    
class Tests(unittest.TestCase):

    def pyomo(self, cmd):
        os.chdir(currdir)
        output = main.run(['-q', '-c']+cmd)
        return output 

class BaselineTests(Tests):
    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)
BaselineTests = unittest.category('smoke', 'nightly','expensive')(BaselineTests)
"""
The following test generates an nl file for the test case
and checks that it matches the current pyomo baseline nl file
"""
@unittest.nottest
def nlwriter_baseline_test(self, name):
    if os.path.exists(currdir+name+'.dat'):
        self.pyomo(['--save-model='+currdir+name+'.test.nl',
                    '--skip-canonical-repn',
                    currdir+name+'_testCase.py',
                    currdir+name+'.dat'])
    else:
        self.pyomo(['--save-model='+currdir+name+'.test.nl',
                    '--skip-canonical-repn',
                    currdir+name+'_testCase.py'])

    # Check that the pyomo nl file matches its own baseline    
    self.assertFileEqualsBaseline(currdir+name+'.test.nl', currdir+name+'.pyomo.nl', tolerance=1e-7)


class ASLTests(Tests):

    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)
ASLTests = unittest.category('smoke','nightly','expensive')(ASLTests)
"""
The following test calls the asl_test executable to generate JSON
files corresponding to both the AMPL-generated nl file and the 
Pyomo-generated nl file. The JSON files are then diffed using
the pyutilib.th test class method assertMatchesJsonBaseline()
"""
@unittest.nottest
def nlwriter_asl_test(self, name):
    if has_asl_test is False:
        self.skipTest('asl_test executable not available')
        return        
    if os.path.exists(currdir+name+'.dat'):
        self.pyomo(['--save-model='+currdir+name+'.test.nl',
                    '--symbolic-solver-labels',
                    '--skip-canonical-repn',
                    currdir+name+'_testCase.py',
                    currdir+name+'.dat'])
    else:
        self.pyomo(['--save-model='+currdir+name+'.test.nl',
                    '--symbolic-solver-labels',
                    '--skip-canonical-repn',
                    currdir+name+'_testCase.py'])

    # compare AMPL and Pyomo nl file structure
    try:
        os.remove(currdir+'stub.json')
    except Exception:
        pass
    try:
        os.remove(currdir+'stub.test.json')
    except Exception:
        pass

    # obtain the nl file summary information for comparison with ampl
    p = pyutilib.subprocess.run('asl_test '+currdir+name+'.test.nl rows='+currdir+name+'.test.row cols='+currdir+name+'.test.col')
    self.assertTrue(p[0] == 0, msg=p[1])
    os.rename(currdir+'stub.json',currdir+'stub.test.json')
    # obtain the nl file summary information for comparison with pyomo
    p = pyutilib.subprocess.run('asl_test '+currdir+name+'.ampl.nl rows='+currdir+name+'.ampl.row cols='+currdir+name+'.ampl.col')
    self.assertTrue(p[0] == 0, msg=p[1])
    self.assertMatchesJsonBaseline(currdir+'stub.test.json', currdir+'stub.json', tolerance=1e-8)
    os.remove(currdir+'stub.json')
    
    # delete temporary test files
    os.remove(currdir+name+'.test.col')
    os.remove(currdir+name+'.test.row')
    os.remove(currdir+name+'.test.nl')


for f in glob.glob(currdir+'*_testCase.py'):
    name = re.split('[._]',os.path.basename(f))[0]
    BaselineTests.add_fn_test(fn=nlwriter_baseline_test, name=name)
    ASLTests.add_fn_test(fn=nlwriter_asl_test, name=name)

if __name__ == "__main__":
    unittest.main()
