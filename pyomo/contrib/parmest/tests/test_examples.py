# This is a test to ensure all of the examples run.
import os
import sys
import unittest
from os import listdir
from os.path import abspath, dirname, isfile, join
from subprocess import call

testdir = dirname(abspath(str(__file__)))
examplesdir = join(testdir, "..", "examples")


class TestExamples(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        import wntr

        self.wntr = wntr

    @classmethod
    def tearDownClass(self):
        pass

    def test_examples(self):
        cwd = os.getcwd()
        os.chdir(examplesdir)
        flag = 0
        failed_examples = []
        for folder in ['rooney_biegler']:
            subdir = join(examplesdir, folder)
            os.chdir(subdir)
            example_files = [
                f
                for f in listdir(subdir)
                if isfile(join(subdir, f))
                and f.endswith(".py")
                and not f.startswith("test")
            ]
            for f in example_files:
                tmp_flag = call([sys.executable, join(subdir, f)])
                print(folder, f, tmp_flag)
                if tmp_flag == 1:
                    failed_examples.append(f)
                    flag = 1
            os.chdir(examplesdir)
        os.chdir(cwd)
        if len(failed_examples) > 0:
            print("failed examples: {0}".format(failed_examples))
        self.assertEqual(flag, 0)


if __name__ == "__main__":
    unittest.main()
