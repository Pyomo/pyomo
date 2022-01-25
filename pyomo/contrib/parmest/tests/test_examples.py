# This is a test to ensure all of the examples run.
# assert statements should be included in the example files
import os
import sys
import unittest
from os import listdir
from os.path import abspath, dirname, isfile, join
import subprocess

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
                for f in listdir(subdir)
                if isfile(join(subdir, f))
                and f.endswith(".py")
            ]
            for f in example_files:
                abspath = os.path.abspath(join(subdir, f))
                ret = subprocess.run([sys.executable, abspath])
                retcode = ret.returncode
                print(folder, f, retcode)
                if retcode == 1:
                    failed_examples.append(abspath)
        os.chdir(cwd)
        if len(failed_examples) > 0:
            print("failed examples: {0}".format(failed_examples))
        self.assertEqual(len(failed_examples), 0)


if __name__ == "__main__":
    unittest.main()
