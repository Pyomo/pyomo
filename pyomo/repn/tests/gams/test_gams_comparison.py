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
# Test the Pyomo BAR writer
#

import re
import glob
import os
from os.path import abspath, dirname, join

from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager

import pyomo.scripting.pyomo_main as main

parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

currdir = this_file_dir()
datadir = abspath(join(currdir, "..", "ampl"))
deleteFiles = True

# add test methods to classes
invalidlist = []
validlist = []

invalid_tests = {'small14'}
for f in glob.glob(join(datadir, '*_testCase.py')):
    name = re.split('[._]', os.path.basename(f))[0]
    if name in invalid_tests:
        # Create some list
        invalidlist.append((name, datadir))
    else:
        validlist.append((name, datadir))

for f in glob.glob(join(currdir, '*_testCase.py')):
    name = re.split('[._]', os.path.basename(f))[0]
    validlist.append((name, currdir))


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
    # The following test generates a GMS file for the test case
    # and checks that it matches the current pyomo baseline GMS file
    #

    @parameterized.parameterized.expand(input=validlist)
    def gams_writer_baseline_test(self, name, targetdir):
        baseline = join(currdir, name + '.pyomo.gms')
        testFile = TempfileManager.create_tempfile(suffix=name + '.test.gms')
        cmd = ['--output=' + testFile, join(targetdir, name + '_testCase.py')]
        if os.path.exists(join(targetdir, name + '.dat')):
            cmd.append(join(targetdir, name + '.dat'))
        self.pyomo(cmd)

        # Check that the pyomo nl file matches its own baseline
        try:
            self.assertTrue(cmp(testFile, baseline))
        except:
            with open(baseline, 'r') as f1, open(testFile, 'r') as f2:
                f1_contents = list(filter(None, f1.read().split()))
                f2_contents = list(filter(None, f2.read().split()))
                self.assertEqual(
                    f1_contents,
                    f2_contents,
                    "\n\nbaseline: %s\ntestFile: %s\n" % (baseline, testFile),
                )

    @parameterized.parameterized.expand(input=invalidlist)
    def gams_writer_test_invalid(self, name, targetdir):
        with self.assertRaisesRegex(
            RuntimeError, "GAMS files cannot represent the unary function"
        ):
            testFile = TempfileManager.create_tempfile(suffix=name + '.test.gms')
            cmd = ['--output=' + testFile, join(targetdir, name + '_testCase.py')]
            if os.path.exists(join(targetdir, name + '.dat')):
                cmd.append(join(targetdir, name + '.dat'))
            self.pyomo(cmd)


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()
