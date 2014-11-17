import os
import sys
from os.path import abspath, dirname, join, basename
import time
import fnmatch
import json

# TODO: test non-trivial bundles for farmer
# TODO: test farmer with integers

import pyutilib.th as unittest
from pyutilib.misc.comparison import open_possibly_compressed_file
import pyutilib.services
from pyomo.pysp.tests.examples.ph_checker import main as validate_ph_main
import pyomo.environ
from pyomo.solvers.tests.io.writer_test_cases import SolverTestCase, testCases

has_yaml = False
try:
    import yaml
    has_yaml = True
except ImportError:
    has_yaml = False

# Global test configuration options
_test_name_wildcard_include = ["*"]
_test_name_wildcard_exclude = [""]
_disable_stdout_test = True
_json_exact_comparison = True
_yaml_exact_comparison = True
_diff_tolerance = 1e-4
_baseline_suffix = ".gz"
_pyro_external_ns = False

#
# Get the directory where this script is defined, and where the baseline
# files are located.
#

thisDir = dirname(abspath(__file__))
baselineDir = join(thisDir,"baselines")
pysp_examples_dir = \
    join(dirname(dirname(dirname(dirname(thisDir)))),"examples","pysp")
pyomo_bin_dir = \
    join(dirname(dirname(dirname(dirname(dirname(dirname(thisDir)))))),"bin")

farmer_examples_dir = join(pysp_examples_dir,"farmer")
farmer_model_dir = join(farmer_examples_dir,"models")
farmer_concrete_model_dir = join(farmer_examples_dir,"concrete")
farmer_expr_model_dir = join(farmer_examples_dir,"expr_models")
farmer_max_model_dir = join(farmer_examples_dir,"maxmodels")
farmer_data_dir = join(farmer_examples_dir,"scenariodata")
farmer_trivialbundlesdata_dir = \
    join(farmer_examples_dir,"scenariodataWithTrivialBundles")
farmer_config_dir = join(farmer_examples_dir,"config")
#farmer_config_dir = join(thisDir, "farmer_files")

nf_examples_dir = join(pysp_examples_dir,"networkflow")
nf_model_dir = join(nf_examples_dir,"models")
nf_data_dir = join(nf_examples_dir,"1ef3")
nf_config_dir = join(nf_examples_dir, "config")

sizes_examples_dir = join(pysp_examples_dir,"sizes")
sizes_model_dir = join(sizes_examples_dir,"models")
sizes_data_dir = join(sizes_examples_dir,"SIZES3")
sizes_config_dir = join(sizes_examples_dir, "config")

forestry_examples_dir = join(pysp_examples_dir,"forestry")
forestry_model_dir = join(forestry_examples_dir,"models-nb-yr")
forestry_data_dir = join(forestry_examples_dir,"unequalProbs")
forestry_config_dir = join(forestry_examples_dir, "config")

testing_solvers = {}
testing_solvers['cplex','lp'] = False
testing_solvers['cplexamp','nl'] = False
testing_solvers['ipopt','nl'] = False
testing_solvers['cplex','python'] = False
testing_solvers['_cplex_persistent','python'] = False
testCases_copy = list(testCases)
testCases_copy.append( SolverTestCase(name='_cplex_persistent',
                                 io='python'))
for test_case in testCases_copy:
    if ((test_case.name,test_case.io) in testing_solvers) and \
       (test_case.available):
        testing_solvers[(test_case.name,test_case.io)] = True

pyutilib.services.register_executable("mpirun")
mpirun_executable = pyutilib.services.registered_executable('mpirun')
mpirun_available = not mpirun_executable is None

def filter_stale_keys(repn):

    if not isinstance(repn, dict):
        return repn
    else:
        return dict((key,filter_stale_keys(value)) \
                    for key,value in repn.items() \
                    if key != 'stale')

def filter_time_and_data_dirs(line):
    return ("seconds" in line) or \
           ("starting at" in line) or \
           ("solve ends" in line) or \
           line.startswith("Output file written to") or \
           ("filename" in line) or \
           ("directory" in line) or \
           ("file" in line) or \
           line.startswith("WARNING:") or \
           line.startswith("Trying to import module=")

# pyro output filtering is complex, due to asynchronous behaviors -
# filter all blather regarding what Pyro components are doing.
def filter_pyro(line):
    if line.startswith("URI") or line.startswith("Object URI") or line.startswith("Dispatcher Object URI") or line.startswith("Dispatcher is ready"):
       return True
    elif line.startswith("Initializing PH"): # added to prevent diff'ing showing up a positive because of PH initialization order relative to the other pyro-based components
        return True
    elif line.startswith("Applying solver"):
       return True
    elif line.startswith("Attempting to find Pyro dispatcher object"):
       return True   
    elif line.startswith("Getting work from"):
       return True
    elif line.startswith("Name Server started."):
       return True   
    elif line.startswith("Name Server gracefully stopped."):
       return True   
    elif line.startswith("Listening for work from"):
       return True
    #elif line.startswith("Error loading pyomo.opt entry point"): # supressing weird error that occasionally pops up when loading plugins
    #   return True
    elif line.startswith("Broadcast server"):
       return True
    elif line.startswith("Failed to locate nameserver - trying again"):
       return True
    elif line.startswith("Failed to find dispatcher object from name server - trying again"):
       return True   
    elif line.startswith("Lost connection to"): # happens when shutting down pyro objects
       return True      
    elif line.startswith("WARNING: daemon bound on hostname that resolves"): # happens when not connected to a network.
       return True      
    elif line.startswith("This is worker") or line.startswith("This is client") or line.startswith("Finding Pyro"):
       return True
    elif line.find("Applying solver") != -1:
       return True
    elif line.find("Solve completed") != -1:
       return True
    #elif line.find("Freeing MIP data") != -1: # CPLEX 12.4 python interface idiocy
    #   return True
    #elif line.find("Warning: integer solution") != -1: # CPLEX 12.4 python interface idiocy
    #   return True
    elif line.startswith("Creating instance"): # non-deterministic order for PH Pyro solver manager
       return True
    elif ("Creating " in line) and (" random bundles using seed=" in line): # non-deterministic order for PH Pyro solver manager
       return True
    elif line.startswith("Exception in thread"): # occasionally raised during Pyro component shutdown
       return True
    elif line.startswith("Traceback"): # occasionally raised during Pyro component shutdown
       return True
    elif line.startswith("File"): # occasionally raised during Pyro component shutdown
       return True               
    return filter_time_and_data_dirs(line)

class PHTester(object):

    baseline_group = None
    num_scenarios = None
    model_directory = None
    instance_directory = None
    solver_manager = None
    solver_name = None
    solver_io = None
    diff_filter = None
    base_command_options = ""

    def setUp(self):
        assert self.baseline_group is not None
        assert self.num_scenarios is not None
        assert self.model_directory is not None
        assert self.instance_directory is not None
        assert self.solver_manager in ('serial','pyro','phpyro')
        assert (self.solver_name,self.solver_io) in testing_solvers
        assert self.diff_filter is not None
        assert self.base_command_options is not None

    def tearDown(self):
        if self.solver_manager in ('pyro','phpyro'):
            self.safe_delete(os.path.join(thisDir,'Pyro_NS_URI'))
        
    @staticmethod
    def safe_delete(filename):
        try:
            os.remove(filename)
        except OSError:
            pass

    def get_cmd_base(self):
        cmd = ''
        cmd += 'cd '+thisDir+'; '
        if self.solver_manager == 'serial':
            cmd += "runph -r 1 --solver-manager=serial"
        elif self.solver_manager == 'pyro':
            cmd += "mpirun "
            if not _pyro_external_ns:
                cmd += "-np 1 pyomo_ns : "
            cmd += "-np 1 dispatch_srvr : "\
                   "-np 1 pyro_mip_server : "\
                   "-np 1 runph -r 1 --solver-manager=pyro --shutdown-pyro"
        elif self.solver_manager == 'phpyro':
            cmd += "mpirun "
            if not _pyro_external_ns:
                cmd += "-np 1 pyomo_ns : "
            cmd += "-np 1 dispatch_srvr : "\
                   "-np %s phsolverserver : "\
                   "-np 1 runph -r 1 --solver-manager=phpyro --shutdown-pyro"\
                   % (self.num_scenarios)
        else:
            raise RuntimeError("Invalid solver manager "+str(self.solver_manager))
        #cmd += " --solver="+self.solver_name
        #cmd += " --solver-io="+self.solver_io
        cmd += " --traceback "
        cmd += self.base_command_options
        return cmd

    @unittest.nottest
    def _baseline_test(self,
                       options_string="",
                       validation_options_string="",
                       cleanup_func=None,
                       rename_func=None,
                       check_baseline_func=None,
                       has_solution_baseline=True,
                       has_stdout_baseline=True):

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
                    "--user-defined-extension=pyomo.pysp.plugins.phhistoryextension "\
                    "--solution-writer=pyomo.pysp.plugins.jsonsolutionwriter "\
                    +options_string+" "\
                    "&> "+join(thisDir,prefix+".out")
        print("Testing command("+basename(prefix)+"): " + argstring)
        self.safe_delete(join(thisDir,prefix+".out"))
        self.safe_delete(join(thisDir,"ph_history.json"))
        self.safe_delete(join(thisDir,prefix+".ph_history.json.out"))
        self.safe_delete(join(thisDir,"ph_solution.json"))
        self.safe_delete(join(thisDir,prefix+".ph_solution.json.out"))
        if cleanup_func is not None:
            cleanup_func(self, class_name, test_name)
        os.system(argstring)
        self.assertTrue(os.path.exists(join(thisDir,"ph_history.json")))
        self.assertTrue(os.path.exists(join(thisDir,"ph_solution.json")))
        os.rename(join(thisDir,"ph_history.json"),
                  join(thisDir,prefix+".ph_history.json.out"))
        os.rename(join(thisDir,"ph_solution.json"),
                  join(thisDir,prefix+".ph_solution.json.out"))
        if rename_func is not None:
            rename_func(self, class_name, test_name)
        group_prefix = self.baseline_group+"."+test_name
        # Disable automatic deletion of the ph_history/ph_solution
        # output file on passing test just in case the optional
        # check_baseline_func wants to look at it.
        validate_ph_main([join(thisDir,prefix+".ph_history.json.out"),
                          '-t',repr(_diff_tolerance)]\
                         +validation_options_string.split())
        if has_solution_baseline:
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(baselineDir,group_prefix+".ph_history.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                delete=False,
                exact=_json_exact_comparison)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(baselineDir,group_prefix+".ph_solution.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                delete=False,
                exact=_json_exact_comparison)
        if check_baseline_func is not None:
            check_baseline_func(self, class_name, test_name)
        else:
            # Now we can safely delete these files because the test
            # has passed if we are here
            self.safe_delete(join(thisDir,prefix+".ph_history.json.out"))
            self.safe_delete(join(thisDir,prefix+".ph_solution.json.out"))

        if (not _disable_stdout_test) and has_stdout_baseline:
            # If the above baseline test passes but this fails, you
            # can assume it is pretty safe to update the following
            # baseline (assuming the output change looks reasonable)
            self.assertFileEqualsBaseline(join(thisDir,prefix+".out"),
                                          join(baselineDir,prefix+".baseline"),
                                          filter=self.diff_filter,
                                          tolerance=_diff_tolerance)
        else:
            self.safe_delete(join(thisDir,prefix+".out"))
        

    @unittest.nottest
    def _phboundextension_baseline_test(self,
                                        options_string="",
                                        validation_options_string="",
                                        cleanup_func=None,
                                        rename_func=None,
                                        check_baseline_func=None,
                                        has_solution_baseline=True,
                                        has_stdout_baseline=True):

        def _cleanup_func(self, class_name, test_name):
            if cleanup_func is not None:
                cleanup_func(self, class_name, test_name)
            prefix = class_name+"."+test_name
            self.safe_delete(join(thisDir,"phbound.txt"))
            self.safe_delete(join(thisDir,"phbestbound.txt"))
            self.safe_delete(join(thisDir,prefix+".phbound.txt.out"))
            self.safe_delete(join(thisDir,prefix+".phbestbound.txt.out"))
        def _rename_func(self, class_name, test_name):
            if rename_func is not None:
                rename_func(self, class_name, test_name)
            prefix = class_name+"."+test_name
            os.rename(join(thisDir,"phbound.txt"),
                      join(thisDir,prefix+".phbound.txt.out"))
            os.rename(join(thisDir,"phbestbound.txt"),
                      join(thisDir,prefix+".phbestbound.txt.out"))
        def _check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            group_prefix = self.baseline_group+"."+test_name
            # Disable automatic deletion of the phbound* output files
            # on passing test just in case the optional
            # check_baseline_func wants to look at them.
            self.assertMatchesYamlBaseline(
                join(thisDir,prefix+".phbound.txt.out"),
                join(baselineDir,group_prefix+".phbound.txt.baseline"),
                tolerance=_diff_tolerance,
                delete=False,
                exact=_yaml_exact_comparison)
            self.assertFileEqualsBaseline(
                join(thisDir,prefix+".phbestbound.txt.out"),
                join(baselineDir,group_prefix+".phbestbound.txt.baseline"),
                tolerance=_diff_tolerance,
                delete=False)
            if check_baseline_func is not None:
                check_baseline_func(self, class_name, test_name)
            else:
                # Now we can safely delete the multiple plugin output
                # files
                self.safe_delete(join(thisDir,prefix+".ph_history.json.out"))
                self.safe_delete(join(thisDir,prefix+".ph_solution.json.out"))
                self.safe_delete(join(thisDir,prefix+".phbound.txt.out"))
                self.safe_delete(join(thisDir,prefix+".phbestbound.txt.out"))


        new_options_string = options_string+(" --user-defined-extension"
                                             "=pyomo.pysp.plugins.phboundextension")
        self._baseline_test(options_string=new_options_string,
                            validation_options_string=validation_options_string,
                            cleanup_func=_cleanup_func,
                            rename_func=_rename_func,
                            check_baseline_func=_check_baseline_func,
                            has_solution_baseline=has_solution_baseline,
                            has_stdout_baseline=has_stdout_baseline)

    @unittest.nottest
    def _convexhullboundextension_baseline_test(self,
                                                options_string="",
                                                validation_options_string="",
                                                cleanup_func=None,
                                                rename_func=None,
                                                check_baseline_func=None,
                                                has_solution_baseline=True,
                                                has_stdout_baseline=True):

        def _cleanup_func(self, class_name, test_name):
            if cleanup_func is not None:
                cleanup_func(self, class_name, test_name)
            prefix = class_name+"."+test_name
            self.safe_delete(join(thisDir,"phbound.txt"))
            self.safe_delete(join(thisDir,"phbestbound.txt"))
            self.safe_delete(join(thisDir,prefix+".phbound.txt.out"))
            self.safe_delete(join(thisDir,prefix+".phbestbound.txt.out"))
        def _rename_func(self, class_name, test_name):
            if rename_func is not None:
                rename_func(self, class_name, test_name)
            prefix = class_name+"."+test_name
            os.rename(join(thisDir,"phbound.txt"),
                      join(thisDir,prefix+".phbound.txt.out"))
            os.rename(join(thisDir,"phbestbound.txt"),
                      join(thisDir,prefix+".phbestbound.txt.out"))
        def _check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            group_prefix = self.baseline_group+"."+test_name
            # Disable automatic deletion of the phbound* output files
            # on passing test just in case the optional
            # check_baseline_func wants to look at them.
            self.assertMatchesYamlBaseline(
                join(thisDir,prefix+".phbound.txt.out"),
                join(baselineDir,group_prefix+".phbound.txt.baseline"),
                tolerance=_diff_tolerance,
                delete=False,
                exact=_yaml_exact_comparison)
            self.assertFileEqualsBaseline(
                join(thisDir,prefix+".phbestbound.txt.out"),
                join(baselineDir,group_prefix+".phbestbound.txt.baseline"),
                tolerance=_diff_tolerance,
                delete=False)
            if check_baseline_func is not None:
                check_baseline_func(self, class_name, test_name)
            else:
                # Now we can safely delete the multiple plugin output files
                self.safe_delete(join(thisDir,prefix+".ph_history.json.out"))
                self.safe_delete(join(thisDir,prefix+".ph_solution.json.out"))
                self.safe_delete(join(thisDir,prefix+".phbound.txt.out"))
                self.safe_delete(join(thisDir,prefix+".phbestbound.txt.out"))

        new_options_string = \
            options_string+(" --user-defined-extension"
                            "=pyomo.pysp.plugins.convexhullboundextension")
        self._baseline_test(options_string=new_options_string,
                            validation_options_string=validation_options_string,
                            cleanup_func=_cleanup_func,
                            rename_func=_rename_func,
                            check_baseline_func=_check_baseline_func,
                            has_solution_baseline=has_solution_baseline,
                            has_stdout_baseline=has_stdout_baseline)

    @unittest.nottest
    def _withef_baseline_test(self,
                              options_string="",
                              validation_options_string="",
                              cleanup_func=None,
                              rename_func=None,
                              check_baseline_func=None,
                              has_solution_baseline=True,
                              has_stdout_baseline=True):

        def _cleanup_func(self, class_name, test_name):
            if cleanup_func is not None:
                cleanup_func(self, class_name, test_name)
            prefix = class_name+"."+test_name
            self.safe_delete(join(thisDir,"postphef_solution.json"))
            self.safe_delete(join(thisDir,prefix+".postphef_solution.json.out"))
        def _rename_func(self, class_name, test_name):
            if rename_func is not None:
                rename_func(self, class_name, test_name)
            prefix = class_name+"."+test_name
            os.rename(join(thisDir,"postphef_solution.json"),
                      join(thisDir,prefix+".postphef_solution.json.out"))
        def _check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            group_prefix = self.baseline_group+"."+test_name
            # Disable automatic deletion of the phbound* output files
            # on passing test just in case the optional
            # check_baseline_func wants to look at them.
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".postphef_solution.json.out"),
                join(baselineDir,group_prefix+".postphef_solution.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                delete=False,
                exact=_json_exact_comparison)
            if check_baseline_func is not None:
                check_baseline_func(self, class_name, test_name)
            else:
                # Now we can safely delete the multiple plugin output files
                self.safe_delete(join(thisDir,prefix+".ph_history.json.out"))
                self.safe_delete(join(thisDir,prefix+".ph_solution.json.out"))
                self.safe_delete(join(thisDir,prefix+".postphef_solution.json.out"))

        new_options_string = options_string+(" --solve-ef")
        self._baseline_test(options_string=new_options_string,
                            validation_options_string=validation_options_string,
                            cleanup_func=_cleanup_func,
                            rename_func=_rename_func,
                            check_baseline_func=_check_baseline_func,
                            has_solution_baseline=has_solution_baseline,
                            has_stdout_baseline=has_stdout_baseline)

    @unittest.nottest
    def _withef_compare_baseline_test(self,
                                      compare_testname,
                                      check_baseline_func=None,
                                      **kwds):

        def _check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            # The phboundextension plugin should not affect
            # the convergence of ph, so we also check
            # ph_history/ph_solution output against the test1 baseline
            group_prefix = self.baseline_group+"."+compare_testname
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(baselineDir,group_prefix+".ph_history.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                exact=_json_exact_comparison)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(baselineDir,group_prefix+".ph_solution.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                exact=_json_exact_comparison)

            if check_baseline_func is not None:
                check_baseline_func(self, class_name, test_name)
            else:
                # Now we can safely delete the multiple plugin output files
                self.safe_delete(join(thisDir,prefix+".ph_history.json.out"))
                self.safe_delete(join(thisDir,prefix+".ph_solution.json.out"))
                self.safe_delete(join(thisDir,prefix+".postphef_solution.json.out"))

        self._withef_baseline_test(check_baseline_func=_check_baseline_func,
                                   **kwds)

class FarmerTester(PHTester):

    def test1(self):
        self._baseline_test()

    def test1_withef(self):
        self._withef_compare_baseline_test("test1")

    def test2(self):
        self._baseline_test(options_string=("--enable-termdiff-convergence "
                                            "--termdiff-threshold=0.01"))

    def test3(self):
        self._baseline_test(options_string="--linearize-nonbinary-penalty-terms=10",
                            validation_options_string="--disable-proximal-term-check")

    def test4(self):
        self._baseline_test(options_string="--retain-quadratic-binary-terms")

    def test5(self):
        self._baseline_test(options_string="--preprocess-fixed-variables")

    # Test the wwphextension plugin (it's best if this test involves
    # variable fixing)
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test6(self):
        self._baseline_test(options_string=("--enable-ww-extensions "
                                            "--ww-extension-cfgfile="
                                            +join(farmer_config_dir,'wwph.cfg')+" "
                                            "--ww-extension-suffixfile="
                                            +join(farmer_config_dir,'wwph.suffixes')))

    # Test the wwphextension plugin (it's best if this test involves
    # variable fixing) and solve ef
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test6_withef(self):
        self._withef_compare_baseline_test("test6",
                                           options_string=("--enable-ww-extensions "
                                                           "--ww-extension-cfgfile="
                                                           +join(farmer_config_dir,'wwph.cfg')+" "
                                                           "--ww-extension-suffixfile="
                                                           +join(farmer_config_dir,'wwph.suffixes')))

    # Test the phboundextension plugin and that it does not effect ph
    # convergence
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test7(self):
        def check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            # The phboundextension plugin should not affect
            # the convergence of ph, so we also check
            # ph_history/ph_solution output against the test1 baseline
            group_prefix = self.baseline_group+".test1"
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(baselineDir,group_prefix+".ph_history.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                exact=_json_exact_comparison)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(baselineDir,group_prefix+".ph_solution.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                exact=_json_exact_comparison)
            # The phbound* output files are not tested
            # against any other baselines. If we made it
            # to this point, they can (and should) be cleaned up
            self.safe_delete(join(thisDir,prefix+".phbound.txt.out"))
            self.safe_delete(join(thisDir,prefix+".phbestbound.txt.out"))
        self._phboundextension_baseline_test(check_baseline_func=check_baseline_func)

    # Test the convexhullboundextension plugin (which does affect ph
    # convergence)
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test8(self):
        self._convexhullboundextension_baseline_test()

    # This is test7 (phboundextension) combined with test6
    # (wwphextension), which involves variable fixing. These plugins
    # should not interact with each other so we additionally test
    # their output files against test6
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test9(self):
        def check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            # The phboundextension plugin should not affect
            # the convergence of ph with the wwphextension plugin
            # enabled, so we also check ph_history/ph_solution output
            # against the test6 baseline
            group_prefix = self.baseline_group+".test6"
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(baselineDir,group_prefix+".ph_history.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                exact=_json_exact_comparison)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(baselineDir,group_prefix+".ph_solution.json.baseline"+_baseline_suffix),
                tolerance=_diff_tolerance,
                exact=_json_exact_comparison)
            # The phbound* output files are not tested against any
            # other baselines. If we made it to this point, they can
            # (and should) be cleaned up
            self.safe_delete(join(thisDir,prefix+".phbound.txt.out"))
            self.safe_delete(join(thisDir,prefix+".phbestbound.txt.out"))
        self._phboundextension_baseline_test(
            options_string=("--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(farmer_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(farmer_config_dir,'wwph.suffixes')),
            check_baseline_func=check_baseline_func)
        
    # This is test8 (convexhullboundextension) combined with test6
    # (wwphextension), which involves variable fixing. These plugins
    # likely interact with each other.  Not sure I can perform any
    # additional tests here.
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test10(self):
        self._convexhullboundextension_baseline_test(
            options_string=("--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(farmer_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(farmer_config_dir,'wwph.suffixes')))

    # This is test9 with the --preprocess-fixed-variables flag
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test11(self):
        def cleanup_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            self.safe_delete(
                join(thisDir,prefix+".ph_history.json.stale_filtered.baseline.out"))
            self.safe_delete(
                join(thisDir,prefix+".ph_solution.json.stale_filtered.baseline.out"))
        def check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            # Test that baselines also match those of test9 without
            # the stale flags (which are flipped for fixed variables)
            group_prefix = self.baseline_group+".test9"
            f = open_possibly_compressed_file(join(baselineDir,
                           group_prefix+".ph_history.json.baseline"+_baseline_suffix))
            baseline_ph_history = json.load(f)
            f.close()
            baseline_ph_history = filter_stale_keys(baseline_ph_history)
            with open(join(thisDir,
                           prefix+\
                           ".ph_history.json.stale_filtered.baseline.out"),'w') as f:
                json.dump(baseline_ph_history,f,indent=2)
            f = open_possibly_compressed_file(join(baselineDir,
                           group_prefix+".ph_solution.json.baseline"+_baseline_suffix))
            baseline_ph_solution = json.load(f)
            f.close()
            baseline_ph_solution = filter_stale_keys(baseline_ph_solution)
            with open(join(thisDir,
                           prefix+\
                           ".ph_solution.json.stale_filtered.baseline.out"),'w') as f:
                json.dump(baseline_ph_solution,f,indent=2)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(thisDir,prefix+".ph_history.json.stale_filtered.baseline.out"),
                tolerance=_diff_tolerance,
                exact=False)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(thisDir,prefix+".ph_solution.json.stale_filtered.baseline.out"),
                tolerance=_diff_tolerance,
                exact=False)
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_history.json.stale_filtered.baseline.out"))
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_solution.json.stale_filtered.baseline.out"))
            self.assertMatchesYamlBaseline(
                join(thisDir,prefix+".phbound.txt.out"),
                join(baselineDir,group_prefix+".phbound.txt.baseline"),
                tolerance=_diff_tolerance,
                exact=_yaml_exact_comparison)
            self.assertFileEqualsBaseline(
                join(thisDir,prefix+".phbestbound.txt.out"),
                join(baselineDir,group_prefix+".phbestbound.txt.baseline"),
                tolerance=_diff_tolerance)
        self._phboundextension_baseline_test(
            options_string=("--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(farmer_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(farmer_config_dir,'wwph.suffixes')+" "
                            "--preprocess-fixed-variables "),
            cleanup_func=cleanup_func,
            check_baseline_func=check_baseline_func)

    # This is test10 with the --preprocess-fixed-variables flag
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test12(self):
        def cleanup_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_history.json.stale_filtered.baseline.out"))
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_solution.json.stale_filtered.baseline.out"))
        def check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            # Test that baselines also match those of test10 without
            # the stale flags (which are flipped for fixed variables)
            group_prefix = self.baseline_group+".test10"
            f = open_possibly_compressed_file(join(baselineDir,
                           group_prefix+".ph_history.json.baseline"+_baseline_suffix))
            baseline_ph_history = json.load(f)
            f.close()
            baseline_ph_history = filter_stale_keys(baseline_ph_history)
            with open(join(thisDir,
                           prefix+\
                           ".ph_history.json.stale_filtered.baseline.out"),'w') as f:
                json.dump(baseline_ph_history,f,indent=2)
            f = open_possibly_compressed_file(join(baselineDir,
                            group_prefix+".ph_solution.json.baseline"+_baseline_suffix))
            baseline_ph_solution = json.load(f)
            f.close()
            baseline_ph_solution = filter_stale_keys(baseline_ph_solution)
            with open(join(thisDir,
                           prefix+\
                           ".ph_solution.json.stale_filtered.baseline.out"),'w') as f:
                json.dump(baseline_ph_solution,f,indent=2)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(thisDir,prefix+".ph_history.json.stale_filtered.baseline.out"),
                tolerance=_diff_tolerance,
                exact=False)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(thisDir,prefix+".ph_solution.json.stale_filtered.baseline.out"),
                tolerance=_diff_tolerance,
                exact=False)
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_history.json.stale_filtered.baseline.out"))
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_solution.json.stale_filtered.baseline.out"))
            self.assertMatchesYamlBaseline(
                join(thisDir,prefix+".phbound.txt.out"),
                join(baselineDir,group_prefix+".phbound.txt.baseline"),
                tolerance=_diff_tolerance,
                exact=_yaml_exact_comparison)
            self.assertFileEqualsBaseline(
                join(thisDir,prefix+".phbestbound.txt.out"),
                join(baselineDir,group_prefix+".phbestbound.txt.baseline"),
                tolerance=_diff_tolerance)

        self._convexhullboundextension_baseline_test(
            options_string=("--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(farmer_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(farmer_config_dir,'wwph.suffixes')+" "
                            "--preprocess-fixed-variables "),
            cleanup_func=cleanup_func,
            check_baseline_func=check_baseline_func)

    # This is test6 with the --preprocess-fixed-variables flag
    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test13(self):
        def cleanup_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_history.json.stale_filtered.baseline.out"))
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_solution.json.stale_filtered.baseline.out"))
        def check_baseline_func(self, class_name, test_name):
            prefix = class_name+"."+test_name
            # Test that baselines also match those of test6 without
            # the stale flags (which are flipped for fixed variables)
            group_prefix = self.baseline_group+".test6"
            f = open_possibly_compressed_file(join(baselineDir,
                           group_prefix+".ph_history.json.baseline"+_baseline_suffix))
            baseline_ph_history = json.load(f)
            f.close()
            baseline_ph_history = filter_stale_keys(baseline_ph_history)
            with open(join(thisDir,
                           prefix+\
                           ".ph_history.json.stale_filtered.baseline.out"),'w') as f:
                json.dump(baseline_ph_history,f,indent=2)
            f = open_possibly_compressed_file(join(baselineDir,
                           group_prefix+\
                           ".ph_solution.json.baseline"+_baseline_suffix))
            baseline_ph_solution = json.load(f)
            f.close()
            baseline_ph_solution = filter_stale_keys(baseline_ph_solution)
            with open(join(thisDir,
                           prefix+\
                           ".ph_solution.json.stale_filtered.baseline.out"),'w') as f:
                json.dump(baseline_ph_solution,f,indent=2)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_history.json.out"),
                join(thisDir,prefix+".ph_history.json.stale_filtered.baseline.out"),
                tolerance=_diff_tolerance,
                exact=False)
            self.assertMatchesJsonBaseline(
                join(thisDir,prefix+".ph_solution.json.out"),
                join(thisDir,prefix+".ph_solution.json.stale_filtered.baseline.out"),
                tolerance=_diff_tolerance,
                exact=False)
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_history.json.stale_filtered.baseline.out"))
            self.safe_delete(join(thisDir,
                                  prefix+\
                                  ".ph_solution.json.stale_filtered.baseline.out"))

        self._baseline_test(
            options_string=("--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(farmer_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(farmer_config_dir,'wwph.suffixes')+" "
                            "--preprocess-fixed-variables "),
            cleanup_func=cleanup_func,
            check_baseline_func=check_baseline_func)

    def test14(self):
        self._baseline_test(
            options_string=("--max-iterations=10 "
                            "--aggregate-cfgfile="
                            +join(farmer_config_dir,'aggregategetter.py')+" "
                            "--rho-cfgfile="
                            +join(farmer_config_dir,'rhosetter.py')+" "
                            "--bounds-cfgfile="
                            +join(farmer_config_dir,'boundsetter.py')))

    def test14_withef(self):
        self._withef_compare_baseline_test(
            "test14",
            options_string=("--max-iterations=10 "
                            "--aggregate-cfgfile="
                            +join(farmer_config_dir,'aggregategetter.py')+" "
                            "--rho-cfgfile="
                            +join(farmer_config_dir,'rhosetter.py')+" "
                            "--bounds-cfgfile="
                            +join(farmer_config_dir,'boundsetter.py')))

#
# create the actual testing classes
#
@unittest.category('expensive')
class TestPHFarmerSerial(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_manager = 'serial'
        cls.solver_name = 'cplexamp'
        cls.solver_io = 'nl'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHFarmerPHPyro(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_manager = 'phpyro'
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHFarmerPyro(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_data_dir
        cls.solver_manager = 'pyro'
        cls.solver_name = 'ipopt'
        cls.solver_io = 'nl'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHFarmerTrivialBundlesSerial(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_trivialbundlesdata_dir
        cls.solver_manager = 'serial'
        cls.solver_name = 'ipopt'
        cls.solver_io = 'nl'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHFarmerTrivialBundlesPHPyro(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_trivialbundlesdata_dir
        cls.solver_manager = 'phpyro'
        cls.solver_name = 'cplex'
        cls.solver_io = 'lp'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHFarmerTrivialBundlesPyro(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = farmer_model_dir
        cls.instance_directory = farmer_trivialbundlesdata_dir
        cls.solver_manager = 'pyro'
        cls.solver_name = 'cplexamp'
        cls.solver_io = 'nl'
        cls.diff_filter = staticmethod(filter_pyro)



@unittest.category('expensive')
class TestPHFarmerSerialPersistent(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = join(farmer_concrete_model_dir,'ReferenceModel.py')
        cls.instance_directory = join(farmer_data_dir,'ScenarioStructure.dat')
        cls.solver_manager = 'serial'
        cls.solver_name = '_cplex_persistent'
        cls.solver_io = 'python'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHFarmerPHPyroPersistent(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = join(farmer_concrete_model_dir,'ReferenceModel.py')
        cls.instance_directory = join(farmer_data_dir,'ScenarioStructure.dat')
        cls.solver_manager = 'phpyro'
        cls.solver_name = '_cplex_persistent'
        cls.solver_io = 'python'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHFarmerTrivialBundlesSerialPersistent(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = join(farmer_concrete_model_dir,'ReferenceModel.py')
        cls.instance_directory = join(farmer_trivialbundlesdata_dir,'ScenarioStructure.dat')
        cls.solver_manager = 'serial'
        cls.solver_name = '_cplex_persistent'
        cls.solver_io = 'python'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHFarmerTrivialBundlesPHPyroPersistent(FarmerTester,unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.baseline_group = "TestPHFarmer"
        cls.num_scenarios = 3
        cls.model_directory = join(farmer_concrete_model_dir,'ReferenceModel.py')
        cls.instance_directory = join(farmer_trivialbundlesdata_dir,'ScenarioStructure.dat')
        cls.solver_manager = 'phpyro'
        cls.solver_name = '_cplex_persistent'
        cls.solver_io = 'python'
        cls.diff_filter = staticmethod(filter_pyro)


class NetworkFlowTester(PHTester):

    @staticmethod
    def _setUpClass(cls):
        cls.baseline_group = "TestPHNetworkFlow1ef3"
        cls.num_scenarios = 3
        cls.model_directory = nf_model_dir
        cls.instance_directory = nf_data_dir
        cls.solver_name = 'cplexamp'
        cls.solver_io = 'nl'

    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test1(self):
        self._baseline_test(
            options_string=("--max-iterations=0 "
                            "--aggregate-cfgfile="
                            +join(nf_config_dir,'aggregategetter.py')+" "
                            "--rho-cfgfile="
                            +join(nf_config_dir,'rhosettermixed.py')+" "
                            "--bounds-cfgfile="
                            +join(nf_config_dir,'xboundsetter.py')+" "
                            "--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(nf_config_dir,'wwph-immediatefixing.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(nf_config_dir,'wwph.suffixes')))

    def test2(self):
        self._baseline_test(
            options_string=("--max-iterations=0 "
                            "--aggregate-cfgfile="
                            +join(nf_config_dir,'aggregategetter.py')+" "
                            "--rho-cfgfile="
                            +join(nf_config_dir,'rhosetter0.5.py')+" "
                            "--bounds-cfgfile="
                            +join(nf_config_dir,'xboundsetter.py')+" "
                            "--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(nf_config_dir,'wwph-aggressivefixing.cfg')))

    def test3(self):
        self._baseline_test(
            options_string=("--max-iterations=0 "
                            "--aggregate-cfgfile="
                            +join(nf_config_dir,'aggregategetter.py')+" "
                            "--rho-cfgfile="
                            +join(nf_config_dir,'rhosetter1.0.py')+" "
                            "--bounds-cfgfile="
                            +join(nf_config_dir,'xboundsetter.py')+" "
                            "--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(nf_config_dir,'wwph-mipgaponly.cfg')))

    def test4(self):
        self._baseline_test(
            options_string=("--max-iterations=1 "
                            "--aggregate-cfgfile="
                            +join(nf_config_dir,'aggregategetter.py')+" "
                            "--rho-cfgfile="
                            +join(nf_config_dir,'rhosetter1.0.py')+" "
                            "--bounds-cfgfile="
                            +join(nf_config_dir,'xboundsetter.py')+" "
                            "--linearize-nonbinary-penalty-terms=20"),
            validation_options_string="--proximal-term-bounds-above",
            has_solution_baseline=False,
            has_stdout_baseline=False)


@unittest.category('expensive')
class TestPHNetworkFlow1ef3Serial(NetworkFlowTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NetworkFlowTester._setUpClass(cls)
        cls.solver_manager = 'serial'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHNetworkFlow1ef3Pyro(NetworkFlowTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NetworkFlowTester._setUpClass(cls)
        cls.solver_manager = 'pyro'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHNetworkFlow1ef3PHPyro(NetworkFlowTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        NetworkFlowTester._setUpClass(cls)
        cls.solver_manager = 'phpyro'
        cls.diff_filter = staticmethod(filter_pyro)

"""
class SizesTester(PHTester):

    @staticmethod
    def _setUpClass(cls):
        cls.baseline_group = "TestPHSizes3"
        cls.num_scenarios = 3
        cls.model_directory = sizes_model_dir
        cls.instance_directory = sizes_data_dir
        cls.solver_name = 'cplexamp'
        cls.solver_io = 'nl'

    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test1(self):
        self._baseline_test(
            options_string=("--max-iterations=0 "
                            "--rho-cfgfile="
                            +join(sizes_config_dir,'rhosetter.py')+" "
                            "--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(sizes_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(sizes_config_dir,'wwph.suffixes')))


@unittest.category('expensive')
class TestPHSizes3Serial(SizesTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SizesTester._setUpClass(cls)
        cls.solver_manager = 'serial'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHSizes3Pyro(SizesTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SizesTester._setUpClass(cls)
        cls.solver_manager = 'pyro'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHSizes3PHPyro(SizesTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        SizesTester._setUpClass(cls)
        cls.solver_manager = 'phpyro'
        cls.diff_filter = staticmethod(filter_pyro)



class ForestryTester(PHTester):

    @staticmethod
    def _setUpClass(cls):
        cls.baseline_group = "TestPHForestryUnequalProbs"
        cls.num_scenarios = 18
        cls.model_directory = forestry_model_dir
        cls.instance_directory = forestry_data_dir
        cls.solver_name = 'cplexamp'
        cls.solver_io = 'nl'

    @unittest.skipIf(not has_yaml, "PyYAML module is not available")
    def test1(self):
        self._baseline_test(
            options_string=("--max-iterations=0 "
                            "--rho-cfgfile="
                            +join(forestry_config_dir,'rhosetter.py')+" "
                            "--bounds-cfgfile="
                            +join(forestry_config_dir,'boundsetter.py')+" "
                            "--enable-ww-extensions "
                            "--ww-extension-cfgfile="
                            +join(forestry_config_dir,'wwph.cfg')+" "
                            "--ww-extension-suffixfile="
                            +join(forestry_config_dir,'wwph-nb.suffixes')))


@unittest.category('expensive')
class TestPHForestryUnequalProbsSerial(ForestryTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ForestryTester._setUpClass(cls)
        cls.solver_manager = 'serial'
        cls.diff_filter = staticmethod(filter_time_and_data_dirs)

@unittest.category('expensive')
class TestPHForestryUnequalProbsPyro(ForestryTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ForestryTester._setUpClass(cls)
        cls.solver_manager = 'pyro'
        cls.diff_filter = staticmethod(filter_pyro)

@unittest.category('expensive')
class TestPHForestryUnequalProbsPHPyro(ForestryTester,unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ForestryTester._setUpClass(cls)
        cls.solver_manager = 'phpyro'
        cls.diff_filter = staticmethod(filter_pyro)
"""

if __name__ == "__main__":

    _disable_stdout_test = False

    import sys
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
    if len(tester.result.failures) or \
            len(tester.result.skipped) or \
            len(tester.result.errors):
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
