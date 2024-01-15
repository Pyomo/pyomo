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
import itertools
import re
import glob
import os
from os.path import abspath, dirname, join

currdir = dirname(abspath(__file__)) + os.sep
datadir = abspath(join(currdir, "..", "ampl")) + os.sep

import pyomo.common.unittest as unittest
import pyomo.common

import pyomo.scripting.pyomo_main as main

parameterized, param_available = pyomo.common.dependencies.attempt_import(
    'parameterized'
)
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')

names = []
# add test methods to classes
for f in itertools.chain(
    glob.glob(join(datadir, '*_testCase.py')), glob.glob(join(currdir, '*_testCase.py'))
):
    names.append(re.split('[._]', os.path.basename(f))[0])


class Tests(unittest.TestCase):
    def pyomo(self, cmd):
        os.chdir(currdir)
        output = main.main(['convert', '--logging=quiet', '-c'] + cmd)
        return output


class BaselineTests(Tests):
    def __init__(self, *args, **kwds):
        Tests.__init__(self, *args, **kwds)

    #
    # The following test generates an BAR file for the test case
    # and checks that it matches the current pyomo baseline BAR file
    #
    @parameterized.parameterized.expand(input=names)
    def barwriter_baseline_test(self, name):
        baseline = join(currdir, name + '.pyomo.bar')
        output = join(currdir, name + '.test.bar')
        if not os.path.exists(baseline):
            self.skipTest("baseline file (%s) not found" % (baseline,))

        if os.path.exists(datadir + name + '_testCase.py'):
            testDir = datadir
        else:
            testDir = currdir
        testCase = testDir + name + '_testCase.py'

        if os.path.exists(testDir + name + '.dat'):
            self.pyomo(['--output=' + output, testCase, testDir + name + '.dat'])
        else:
            self.pyomo(['--output=' + output, testCase])

        # Check that the pyomo BAR file matches its own baseline
        with open(baseline, 'r') as f1, open(output, 'r') as f2:
            f1_contents = list(filter(None, f1.read().split()))
            f2_contents = list(filter(None, f2.read().split()))
            for item1, item2 in itertools.zip_longest(f1_contents, f2_contents):
                try:
                    self.assertAlmostEqual(float(item1), float(item2))
                except:
                    self.assertEqual(
                        item1,
                        item2,
                        "\n\nbaseline: %s\ntestFile: %s\n" % (baseline, output),
                    )
        os.remove(join(currdir, name + '.test.bar'))


if __name__ == "__main__":
    unittest.main()
