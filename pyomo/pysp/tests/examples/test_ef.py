#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import fnmatch
import os
from os.path import abspath, dirname, join, basename

try:
    from subprocess import check_output as _run_cmd
except:
    # python 2.6
    from subprocess import check_call as _run_cmd

import pyutilib.th as unittest

from pyomo.pysp.tests.examples.ef_checker import main as validate_ef_main

# Global test configuration options
_test_name_wildcard_include = ["*"]
_test_name_wildcard_exclude = [""]
_disable_stdout_test = True
_json_exact_comparison = True
_yaml_exact_comparison = True
_diff_tolerance = 1e-5

#
# Get the directory where this script is defined, and where the baseline
# files are located.
#

thisDir = dirname(abspath(__file__))
baselineDir = join(thisDir,"baselines")
pysp_examples_dir = join(dirname(dirname(dirname(dirname(thisDir)))),"examples","pysp")
pyomo_bin_dir = join(dirname(dirname(dirname(dirname(dirname(dirname(thisDir)))))),"bin")

farmer_examples_dir = join(pysp_examples_dir,"farmer")
farmer_model_dir = join(farmer_examples_dir,"models")
farmer_concrete_model_dir = join(farmer_examples_dir,"concrete")
farmer_expr_model_dir = join(farmer_examples_dir,"expr_models")
farmer_max_model_dir = join(farmer_examples_dir,"maxmodels")
farmer_expr_max_model_dir = join(farmer_examples_dir,"expr_maxmodels")
farmer_data_dir = join(farmer_examples_dir,"scenariodata")
farmer_trivialbundlesdata_dir = join(farmer_examples_dir,"scenariodataWithTrivialBundles")
farmer_other_dir = join(thisDir, "farmer_files")

forestry_examples_dir = join(pysp_examples_dir,"forestry")
forestry_model_dir = join(forestry_examples_dir,"models-nb-yr")
forestry_expr_model_dir = join(forestry_examples_dir,"expr-models-nb-yr")
forestry_data_dir = join(forestry_examples_dir,"18scenarios")
forestry_unequal_data_dir = join(forestry_examples_dir,"unequalProbs")

hydro_examples_dir = join(pysp_examples_dir,"hydro")
hydro_model_dir = join(hydro_examples_dir,"models")
hydro_data_dir = join(hydro_examples_dir,"scenariodata")
hydro_nodedata_dir = join(hydro_examples_dir,"nodedata")

testing_solvers = {}
testing_solvers['cplex','lp'] = False
testing_solvers['cplex','nl'] = False
testing_solvers['ipopt','nl'] = False
testing_solvers['cplex','python'] = False

class EFTester(object):

    baseline_group = None
    model_directory = None
    instance_directory = None
    solver_name = None
    solver_io = None
    base_command_options = ""

    @staticmethod
    def _setUpClass(cls):
        global testing_solvers
        from pyomo.solvers.tests.solvers import test_solver_cases
        for _solver, _io in test_solver_cases():
            if (_solver, _io) in testing_solvers and \
                test_solver_cases(_solver, _io).available:
                testing_solvers[_solver, _io] = True

    def setUp(self):
        assert self.model_directory is not None
        assert self.instance_directory is not None
        assert (self.solver_name,self.solver_io) in testing_solvers
        assert self.base_command_options is not None

    @staticmethod
    def safe_delete(filename):
        try:
            os.remove(filename)
        except OSError:
            pass

    def get_cmd_base(self):
        cmd = ''
        cmd += 'cd '+thisDir+'; '
        cmd += "runef --solve"
        cmd += " --solver="+self.solver_name
        cmd += " --solver-io="+self.solver_io
        cmd += " --traceback "
        cmd += self.base_command_options
        return cmd

    @unittest.nottest
    def _baseline_test(self,
                       options_string="",
                       validation_options_string="",
                       cleanup_func=None,
                       rename_func=None,
                       check_baseline_func=None):

        global _test_name_wildcard_include
        global _test_name_wildcard_exclude
        class_name, test_name = self.id().split('.')[-2:]
        if all(not fnmatch.fnmatch(class_name+'.'+test_name,inc_wildcard) \
               for inc_wildcard in _test_name_wildcard_include):
            self.skipTest("Test %s.%s does not match any include wildcard in '%s'" \
                          % (class_name, test_name, _test_name_wildcard_include))
        if any(fnmatch.fnmatch(class_name+'.'+test_name,exc_wildcard) \
               for exc_wildcard in _test_name_wildcard_exclude):
            self.skipTest("Test %s.%s matches at least one exclude wildcard in '%s'" \
                          % (class_name, test_name, _test_name_wildcard_exclude))
        if not testing_solvers[self.solver_name,self.solver_io]:
            self.skipTest("Solver (%s,%s) not available"
                          % (self.solver_name, self.solver_io))
        prefix = class_name+"."+test_name
        argstring = self.get_cmd_base()+" "\
                    "--model-directory="+self.model_directory+" "\
                    "--instance-directory="+self.instance_directory+" "\
                    "--solution-writer=pyomo.pysp.plugins.jsonsolutionwriter "\
                    +options_string+" "\
                    "&> "+join(thisDir,prefix+".out")
        print("Testing command("+basename(prefix)+"): " + argstring)
        self.safe_delete(join(thisDir,prefix+".out"))
        self.safe_delete(join(thisDir,"ef_solution.json"))
        self.safe_delete(join(thisDir,prefix+".ef_solution.json.out"))
        if cleanup_func is not None:
            cleanup_func(self, class_name, test_name)
        _run_cmd(argstring, shell=True)
        self.assertTrue(os.path.exists(join(thisDir,"ef_solution.json")))
        os.rename(join(thisDir,"ef_solution.json"),
                  join(thisDir,prefix+".ef_solution.json.out"))
        if rename_func is not None:
            rename_func(self, class_name, test_name)

        validate_ef_main([join(thisDir,prefix+".ef_solution.json.out"),
                          '-t',repr(_diff_tolerance)]\
                         +validation_options_string.split())

        if self.baseline_group is not None:
            group_prefix = self.baseline_group+"."+test_name
            # Disable automatic deletion of the ef_solution output
            # file on passing test just in case the optional
            # check_baseline_func wants to look at it.
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ef_solution.json.out"),
                join(baselineDir,group_prefix+".ef_solution.json.baseline.gz"),
                tolerance=_diff_tolerance,
                delete=False,
                exact=_json_exact_comparison)

        if check_baseline_func is not None:
            assert self.baseline_group is not None
            check_baseline_func(self, class_name, test_name)
        else:
            # Now we can safely delete this file because the test has
            # passed if we are here
            self.safe_delete(join(thisDir,prefix+".ef_solution.json.out"))
        self.safe_delete(join(thisDir,prefix+".out"))

    def test1(self):
        self._baseline_test()

    def test2(self):
        self._baseline_test(options_string="--generate-weighted-cvar")

    def test3(self):
        self._baseline_test(options_string="--generate-weighted-cvar "
                                           "--cvar-weight=0 "
                                           "--risk-alpha=0.1")

    def test4(self):
        self._baseline_test(options_string="--generate-weighted-cvar "
                                           "--cvar-weight=0.5")

    def test5(self):
        self._baseline_test(options_string="--generate-weighted-cvar "
                                           "--risk-alpha=0.1")

@unittest.category('expensive')
class TestEFFarmerCPLEXNL(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = "TestEFFarmer"
        cls.model_directory = join(farmer_concrete_model_dir,'ReferenceModel.py')
        cls.instance_directory = join(farmer_data_dir,'ScenarioStructure.dat')
        cls.solver_name = 'cplex'
        cls.solver_io = 'nl'

@unittest.category('expensive')
class TestEFFarmerCPLEXLP(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = "TestEFFarmer"
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'

@unittest.category('expensive')
class TestEFFarmerCPLEXPYTHON(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = "TestEFFarmer"
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'python'

@unittest.category('expensive')
class TestEFFarmerExpression(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = "TestEFFarmer"
        cls.model_directory = farmer_expr_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'

@unittest.category('expensive')
class TestEFFarmerMax(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = "TestEFFarmerMax"
        cls.model_directory = farmer_max_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'

@unittest.category('expensive')
class TestEFFarmerMaxExpression(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = "TestEFFarmerMax"
        cls.model_directory = farmer_expr_max_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'






@unittest.category('expensive')
class TestEFForestryNoBaseline(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = None
        cls.model_directory = forestry_model_dir
        cls.instance_directory = forestry_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'
        cls.base_command_options = "--mipgap=0.0125"

@unittest.category('expensive')
class TestEFForestryExpressionNoBaseline(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = None
        cls.model_directory = forestry_expr_model_dir
        cls.instance_directory = forestry_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'
        cls.base_command_options = "--mipgap=0.0125"

@unittest.category('expensive')
class TestEFForestryUnequalProbsNoBaseline(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = None
        cls.model_directory = forestry_model_dir
        cls.instance_directory = forestry_unequal_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'
        cls.base_command_options = "--mipgap=0.0125"

@unittest.category('expensive')
class TestEFForestryExpressionUnequalProbsNoBaseline(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = None
        cls.model_directory = forestry_expr_model_dir
        cls.instance_directory = forestry_unequal_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'
        cls.base_command_options = "--mipgap=0.0125"



@unittest.category('expensive')
class TestEFHydroNoBaseline(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = None
        cls.model_directory = hydro_model_dir
        cls.instance_directory = hydro_data_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'

@unittest.category('expensive')
class TestEFHydroNodeDataNoBaseline(EFTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        EFTester._setUpClass(cls)
        cls.baseline_group = None
        cls.model_directory = hydro_model_dir
        cls.instance_directory = hydro_nodedata_dir
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'


if __name__ == "__main__":

    _disable_stdout_test = False

    if '--include' in sys.argv:
        _test_name_wildcard_include = []
        while '--include' in sys.argv:
            idx = sys.argv.index('--include')
            _test_name_wildcard_include.append(sys.argv[idx+1])
            sys.argv.remove('--include')
            sys.argv.remove(_test_name_wildcard_include[-1])
    if '--exclude' in sys.argv:
        _test_name_wildcard_exclude = []
        while '--exclude' in sys.argv:
            idx = sys.argv.index('--exclude')
            _test_name_wildcard_exclude.append(sys.argv[idx+1])
            sys.argv.remove('--exclude')
            sys.argv.remove(_test_name_wildcard_exclude[-1])
    if '--disable-stdout-test' in sys.argv:
        sys.argv.remove('--disable-stdout-test')
        _disable_stdout_test = True

    print("Including all tests matching wildcard: '%s'" % _test_name_wildcard_include)
    print("Excluding all tests matching wildcard: '%s'" % _test_name_wildcard_exclude)

    tester = unittest.main(exit=False)
    if len(tester.result.failures) or len(tester.result.skipped) or len(tester.result.errors):
        with open('UnitTestNoPass.txt','w') as f:
            f.write("Failures:\n")
            for res in tester.result.failures:
                f.write('.'.join(res[0].id().split('.')[-2:])+' ')
            f.write("\n\nSkipped:\n")
            for res in tester.result.skipped:
                f.write('.'.join(res[0].id().split('.')[-2:])+' ')
            f.write("\n\nErrors:\n")
            for res in tester.result.errors:
                f.write('.'.join(res[0].id().split('.')[-2:])+' ')
