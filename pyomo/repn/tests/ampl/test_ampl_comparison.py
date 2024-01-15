#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Test the Pyomo NL writer against the AMPL NL writer
#

from itertools import zip_longest
import json
import re
import glob
import subprocess
import os
from os.path import join

import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

import pyomo.scripting.pyomo_main as main

parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

currdir = this_file_dir()
deleteFiles = True

# https://github.com/ghackebeil/gjh_asl_json
has_gjh_asl_json = False
if os.system('gjh_asl_json -v') == 0:
    has_gjh_asl_json = True

names = []
# add test methods to classes
for f in glob.glob(join(currdir, '*_testCase.py')):
    names.append(re.split('[._]', os.path.basename(f))[0])


class Tests(unittest.TestCase):
    def pyomo(self, cmd):
        os.chdir(currdir)
        output = main.main(['convert', '--logging=quiet', '-c'] + cmd)
        return output

    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())


class BaselineTests(Tests):
    #
    # The following test generates an nl file for the test case
    # and checks that it matches the current pyomo baseline nl file
    #

    @parameterized.parameterized.expand(input=names)
    def nlwriter_baseline_test(self, name):
        baseline = join(currdir, name + '.pyomo.nl')
        testFile = TempfileManager.create_tempfile(suffix=name + '.test.nl')
        cmd = ['--output=' + testFile, join(currdir, name + '_testCase.py')]
        if os.path.exists(join(currdir, name + '.dat')):
            cmd.append(join(currdir, name + '.dat'))
        self.pyomo(cmd)

        # Check that the pyomo nl file matches its own baseline
        with open(testFile, 'r') as f1, open(baseline, 'r') as f2:
            f1_contents = list(filter(None, f1.read().replace('n', 'n ').split()))
            f2_contents = list(filter(None, f2.read().replace('n', 'n ').split()))
            for item1, item2 in zip_longest(f1_contents, f2_contents):
                try:
                    self.assertEqual(float(item1), float(item2))
                except:
                    self.assertEqual(item1, item2)


@unittest.skipUnless(has_gjh_asl_json, "'gjh_asl_json' executable not available")
class ASLJSONTests(Tests):
    #
    # The following test calls the gjh_asl_json executable to
    # generate JSON files corresponding to both the
    # AMPL-generated nl file and the Pyomo-generated nl
    # file. The JSON files are then diffed using the pyomo.common.unittest
    # test class method assertStructuredAlmostEqual
    #
    @parameterized.parameterized.expand(input=names)
    def nlwriter_asl_test(self, name):
        testFile = TempfileManager.create_tempfile(suffix=name + '.test.nl')
        testFile_row = testFile[:-2] + 'row'
        TempfileManager.add_tempfile(testFile_row, exists=False)
        testFile_col = testFile[:-2] + 'col'
        TempfileManager.add_tempfile(testFile_col, exists=False)

        cmd = [
            '--output=' + testFile,
            '--file-determinism=2',
            '--symbolic-solver-labels',
            join(currdir, name + '_testCase.py'),
        ]
        if os.path.exists(join(currdir, name + '.dat')):
            cmd.append(join(currdir, name + '.dat'))

        self.pyomo(cmd)

        #
        # compare AMPL and Pyomo nl file structure
        #
        testFile_json = testFile[:-2] + 'json'
        TempfileManager.add_tempfile(testFile_json, exists=False)
        # obtain the nl file summary information for comparison with ampl
        p = subprocess.run(
            [
                'gjh_asl_json',
                testFile,
                'rows=' + testFile_row,
                'cols=' + testFile_col,
                'json=' + testFile_json,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertTrue(p.returncode == 0, msg=p.stdout)

        baseFile = join(currdir, name + '.ampl.nl')
        amplFile = TempfileManager.create_tempfile(suffix=name + '.ampl.json')
        # obtain the nl file summary information for comparison with ampl
        p = subprocess.run(
            [
                'gjh_asl_json',
                baseFile,
                'rows=' + baseFile[:-2] + 'row',
                'cols=' + baseFile[:-2] + 'col',
                'json=' + amplFile,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertTrue(p.returncode == 0, msg=p.stdout)

        with open(testFile_json, 'r') as f1, open(amplFile, 'r') as f2:
            self.assertStructuredAlmostEqual(json.load(f1), json.load(f2), abstol=1e-8)


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()
