import os
import sys
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyomo.core
import pyutilib.th as unittest
import pyutilib.subprocess
try:
    import yaml
    yaml_available=True
except ImportError:
    yaml_available=False

def filter1(line):
    return False

def filter2(line):
    return 'Id' in line


@unittest.skipIf(not yaml_available, "PyYaml is not installed")
@unittest.skipIf(not pyomo.common.Executable("glpsol"),
                 "The 'glpsol' executable is not available")
class Test(unittest.TestCase):

    def setUp(self):
        self.cwd = currdir
        os.chdir(self.cwd)

    def tearDown(self):
        os.chdir(self.cwd)

    def run_script(self, test, filter, yaml=False):
        cwd = self.cwd+os.sep+test+os.sep
        os.chdir(cwd)
        pyutilib.subprocess.run(sys.executable+' script.py', outfile=cwd+os.sep+'script.log', cwd=cwd)
        if yaml:
            self.assertMatchesYamlBaseline(cwd+"script.log", cwd+"script.out", tolerance=1e-3)
        else:
            self.assertFileEqualsBaseline(cwd+"script.log", cwd+"script.out", tolerance=1e-3, filter=filter)

    def test_s1(self):
        self.run_script('s1', filter1, True)

    def test_s2(self):
        self.run_script('s2', filter2)


if __name__ == "__main__":
    unittest.main()
