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
# Test the Pyomo command-line interface
#

from itertools import zip_longest
import json
import re
import os
import sys
from os.path import abspath, dirname, join
from filecmp import cmp
import subprocess
import pyomo.common.unittest as unittest

from pyomo.common.dependencies import yaml_available
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.core
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers

from io import StringIO

currdir = this_file_dir()

_diff_tol = 1e-6

deleteFiles = True

solvers = None


class BaseTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global solvers
        import pyomo.environ

        solvers = check_available_solvers('glpk')

    def pyomo(self, cmd, **kwds):
        if 'root' in kwds:
            OUTPUT = kwds['root'] + '.out'
            results = kwds['root'] + '.jsn'
            TempfileManager.add_tempfile(OUTPUT, exists=False)
            TempfileManager.add_tempfile(results, exists=False)
        else:
            OUTPUT = StringIO()
            results = 'results.jsn'
            TempfileManager.create_tempfile(suffix='results.jsn')
        with capture_output(OUTPUT):
            try:
                _dir = os.getcwd()
                os.chdir(currdir)
                args = [
                    'solve',
                    '--solver=glpk',
                    '--results-format=json',
                    '--save-results=%s' % results,
                ]
                if type(cmd) is list:
                    args.extend(cmd)
                elif cmd.endswith('json') or cmd.endswith('yaml'):
                    args.append(cmd)
                else:
                    args.extend(re.split('[ ]+', cmd))
                output = main.main(args)
            finally:
                os.chdir(_dir)
        if not 'root' in kwds:
            return OUTPUT.getvalue()
        return output

    def setUp(self):
        if not 'glpk' in solvers:
            self.skipTest("GLPK is not installed")
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=deleteFiles or self.currentTestPassed())

    def run_pyomo(self, cmd, root):
        results = root + '.jsn'
        TempfileManager.add_tempfile(results, exists=False)
        output = root + '.out'
        TempfileManager.add_tempfile(output, exists=False)
        cmd = [
            'pyomo',
            'solve',
            '--solver=glpk',
            '--results-format=json',
            '--save-results=%s' % results,
        ] + cmd
        with open(output, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=f)
        return result


class TestJson(BaseTester):
    def compare_json(self, file1, file2):
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            f1_contents = json.load(f1)
            f2_contents = json.load(f2)
            self.assertStructuredAlmostEqual(
                f2_contents, f1_contents, abstol=_diff_tol, allow_second_superset=True
            )

    def filter_items(self, items):
        filtered = []
        for i in items:
            if not (i.startswith('/') or i.startswith(":\\", 1)):
                try:
                    filtered.append(float(i))
                except:
                    filtered.append(i)
        return filtered

    def compare_files(self, file1, file2):
        try:
            self.assertTrue(
                cmp(file1, file2), msg="Files %s and %s differ" % (file1, file2)
            )
        except:
            with open(file1, 'r') as f1, open(file2, 'r') as f2:
                f1_contents = f1.read().strip().split('\n')
                f2_contents = f2.read().strip().split('\n')
                f1_filtered = []
                f2_filtered = []
                for item1, item2 in zip_longest(f1_contents, f2_contents):
                    if not item1:
                        f1_filtered.append(item1)
                    elif not item1.startswith('['):
                        items1 = item1.strip().split()
                        f1_filtered.append(self.filter_items(items1))
                    if not item2:
                        f2_filtered.append(item2)
                    elif not item2.startswith('['):
                        items2 = item2.strip().split()
                        f2_filtered.append(self.filter_items(items2))
                self.assertStructuredAlmostEqual(
                    f2_filtered, f1_filtered, abstol=1e-6, allow_second_superset=True
                )

    def test1_simple_pyomo_execution(self):
        # Simple execution of 'pyomo'
        self.pyomo(
            [join(currdir, 'pmedian.py'), join(currdir, 'pmedian.dat')],
            root=join(currdir, 'test1'),
        )
        self.compare_json(join(currdir, 'test1.jsn'), join(currdir, 'test1.txt'))

    def test1a_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' in a subprocess
        files = [
            os.path.join(currdir, 'pmedian.py'),
            os.path.join(currdir, 'pmedian.dat'),
        ]
        self.run_pyomo(files, root=os.path.join(currdir, 'test1a'))
        self.compare_json(join(currdir, 'test1a.jsn'), join(currdir, 'test1.txt'))

    def test1b_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with a configuration file
        self.pyomo(join(currdir, 'test1b.json'), root=join(currdir, 'test1'))
        self.compare_json(join(currdir, 'test1.jsn'), join(currdir, 'test1.txt'))

    def test2_bad_model_name(self):
        # Run pyomo with bad --model-name option value
        self.pyomo(
            '--model-name=dummy pmedian.py pmedian.dat', root=join(currdir, 'test2')
        )
        self.compare_files(join(currdir, "test2.out"), join(currdir, "test2.txt"))

    def test2b_bad_model_name(self):
        # Run pyomo with bad --model-name option value (configfile)
        self.pyomo(join(currdir, 'test2b.json'), root=join(currdir, 'test2'))
        self.compare_files(join(currdir, "test2.out"), join(currdir, "test2.txt"))

    def test3_missing_model_object(self):
        # Run pyomo with model that does not define model object
        self.pyomo('pmedian1.py pmedian.dat', root=join(currdir, 'test3'))
        self.compare_json(join(currdir, "test3.jsn"), join(currdir, "test1.txt"))

    def test4_valid_modelname_option(self):
        # Run pyomo with good --model-name option value
        self.pyomo(
            '--model-name=MODEL ' + join(currdir, 'pmedian1.py pmedian.dat'),
            root=join(currdir, 'test4'),
        )
        self.compare_json(join(currdir, "test4.jsn"), join(currdir, "test1.txt"))

    def test4b_valid_modelname_option(self):
        # Run pyomo with good 'object name' option value (configfile)
        self.pyomo(join(currdir, 'test4b.json'), root=join(currdir, 'test4b'))
        self.compare_json(join(currdir, "test4b.jsn"), join(currdir, "test1.txt"))

    def test5_create_model_fcn(self):
        # """Run pyomo with create_model function"""
        self.pyomo('pmedian2.py pmedian.dat', root=join(currdir, 'test5'))
        self.compare_files(join(currdir, "test5.out"), join(currdir, "test5.txt"))

    def test5b_create_model_fcn(self):
        # Run pyomo with create_model function (configfile)
        self.pyomo(join(currdir, 'test5b.json'), root=join(currdir, 'test5'))
        self.compare_files(join(currdir, "test5.out"), join(currdir, "test5.txt"))

    def test8_instanceonly_option(self):
        # """Run pyomo with --instance-only option"""
        output = self.pyomo(
            '--instance-only pmedian.py pmedian.dat', root=join(currdir, 'test8')
        )
        self.assertEqual(type(output.retval.instance), pyomo.core.ConcreteModel)
        # Check that the results file was NOT created
        self.assertFalse(os.path.exists(join(currdir, 'test8.jsn')))

    def test8b_instanceonly_option(self):
        # Run pyomo with --instance-only option (configfile)
        output = self.pyomo(join(currdir, 'test8b.json'), root=join(currdir, 'test8'))
        self.assertEqual(type(output.retval.instance), pyomo.core.ConcreteModel)
        # Check that the results file was NOT created
        self.assertFalse(os.path.exists(join(currdir, 'test8.jsn')))

    def test9_disablegc_option(self):
        # """Run pyomo with --disable-gc option"""
        output = self.pyomo(
            '--disable-gc pmedian.py pmedian.dat', root=join(currdir, 'test9')
        )
        self.assertEqual(type(output.retval.instance), pyomo.core.ConcreteModel)

    def test9b_disablegc_option(self):
        # Run pyomo with --disable-gc option (configfile)
        output = self.pyomo(join(currdir, 'test9b.json'), root=join(currdir, 'test9'))
        self.assertEqual(type(output.retval.instance), pyomo.core.ConcreteModel)

    def test12_output_option(self):
        # """Run pyomo with --output option"""
        log = join(currdir, 'test12.log')
        TempfileManager.add_tempfile(log, exists=False)
        self.pyomo(
            '--logfile=%s pmedian.py pmedian.dat' % (log,), root=join(currdir, 'test12')
        )
        self.compare_json(join(currdir, "test12.jsn"), join(currdir, "test12.txt"))

    def test12b_output_option(self):
        # Run pyomo with --output option (configfile)
        log = join(currdir, 'test12b.log')
        TempfileManager.add_tempfile(log, exists=False)
        self.pyomo(join(currdir, 'test12b.json'), root=join(currdir, 'test12'))
        self.compare_json(join(currdir, "test12.jsn"), join(currdir, "test12.txt"))

    def test14_concrete_model_with_constraintlist(self):
        # Simple execution of 'pyomo' with a concrete model and constraint lists
        self.pyomo('pmedian4.py', root=join(currdir, 'test14'))
        self.compare_json(join(currdir, "test14.jsn"), join(currdir, "test14.txt"))

    def test14b_concrete_model_with_constraintlist(self):
        # Simple execution of 'pyomo' with a concrete model and constraint lists (configfile)
        self.pyomo('pmedian4.py', root=join(currdir, 'test14'))
        self.compare_json(join(currdir, "test14.jsn"), join(currdir, "test14.txt"))

    def test15_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(
            [
                '--solver-options="mipgap=0.02 cuts="',
                join(currdir, 'pmedian.py'),
                'pmedian.dat',
            ],
            root=join(currdir, 'test15'),
        )
        self.compare_json(join(currdir, "test15.jsn"), join(currdir, "test1.txt"))

    def test15b_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(join(currdir, 'test15b.json'), root=join(currdir, 'test15b'))
        self.compare_json(join(currdir, "test15b.jsn"), join(currdir, "test1.txt"))

    def test15c_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(join(currdir, 'test15c.json'), root=join(currdir, 'test15c'))
        self.compare_json(join(currdir, "test15c.jsn"), join(currdir, "test1.txt"))


@unittest.skipIf(not yaml_available, "YAML not available available")
class TestWithYaml(BaseTester):
    def compare_json(self, file1, file2):
        with open(file1, 'r') as f1, open(file2, 'r') as f2:
            f1_contents = json.load(f1)
            f2_contents = json.load(f2)
            self.assertStructuredAlmostEqual(
                f2_contents, f1_contents, abstol=_diff_tol, allow_second_superset=True
            )

    def test15b_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(join(currdir, 'test15b.yaml'), root=join(currdir, 'test15b'))
        self.compare_json(join(currdir, "test15b.jsn"), join(currdir, "test1.txt"))

    def test15c_simple_pyomo_execution(self):
        # Simple execution of 'pyomo' with options
        self.pyomo(join(currdir, 'test15c.yaml'), root=join(currdir, 'test15c'))
        self.compare_json(join(currdir, "test15c.jsn"), join(currdir, "test1.txt"))


if __name__ == "__main__":
    deleteFiles = False
    unittest.main()
