#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# This is a test to ensure all of the parmest examples run.
# assert statements should be included in the example files

import platform
is_osx = platform.mac_ver()[0] != ''

import pyomo.common.unittest as unittest
import sys
import os
import subprocess
from os.path import abspath, dirname, isfile, join

testdir = dirname(abspath(str(__file__)))
examplesdir = join(testdir, "..", "examples")


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def test_examples(self):
        cwd = os.getcwd()
        os.chdir(examplesdir)
        failed_examples = []
        for folder in ['rooney_biegler']:
            subdir = join(examplesdir, folder)
            os.chdir(subdir)
            example_files = [
                f
                for f in os.listdir(subdir)
                if isfile(join(subdir, f))
                and f.endswith(".py")
            ]
            for f in example_files:
                file_abspath = abspath(join(subdir, f))
                ret = subprocess.run([sys.executable, file_abspath])
                retcode = ret.returncode
                print(folder, f, retcode)
                if retcode == 1:
                    failed_examples.append(file_abspath)
        os.chdir(cwd)
        if len(failed_examples) > 0:
            print("failed examples: {0}".format(failed_examples))
        self.assertEqual(len(failed_examples), 0)


if __name__ == "__main__":
    unittest.main()
