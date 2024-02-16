#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from filecmp import cmp
import os
import sys
import subprocess
from os.path import abspath, dirname

currdir = dirname(abspath(__file__)) + os.sep

import pyomo.core
import pyomo.common.unittest as unittest

try:
    import yaml

    yaml_available = True
except ImportError:
    yaml_available = False


@unittest.skipIf(not yaml_available, "PyYaml is not installed")
@unittest.skipIf(
    not pyomo.common.Executable("glpsol"), "The 'glpsol' executable is not available"
)
class Test(unittest.TestCase):
    def setUp(self):
        self.cwd = currdir
        os.chdir(self.cwd)

    def tearDown(self):
        os.chdir(self.cwd)

    def run_script(self, test, yaml_available=False):
        cwd = self.cwd + os.sep + test + os.sep
        os.chdir(cwd)
        with open(cwd + os.sep + 'script.log', 'w') as f:
            subprocess.run([sys.executable, 'script.py'], stdout=f, stderr=f, cwd=cwd)
        if yaml_available:
            with open(cwd + 'script.log', 'r') as f1:
                with open(cwd + 'script.out', 'r') as f2:
                    baseline = yaml.full_load(f1)
                    output = yaml.full_load(f2)
                    self.assertStructuredAlmostEqual(
                        output, baseline, allow_second_superset=True
                    )
        else:
            _log = os.path.join(cwd, 'script.log')
            _out = os.path.join(cwd, 'script.out')
            self.assertTrue(
                cmp(_log, _out), msg="Files %s and %s differ" % (_log, _out)
            )

    def test_s1(self):
        self.run_script('s1', True)

    def test_s2(self):
        self.run_script('s2')


if __name__ == "__main__":
    unittest.main()
