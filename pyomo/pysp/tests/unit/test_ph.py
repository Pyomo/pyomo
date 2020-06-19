#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Get the directory where this script is defined, and where the baseline
# files are located.
#

import os
import sys
import subprocess
from os.path import abspath, dirname

try:
    from subprocess import check_output as _run_cmd
except:
    # python 2.6
    from subprocess import check_call as _run_cmd

thisdir = dirname(abspath(__file__))
this_test_file_directory = dirname(abspath(__file__))+os.sep

pysp_examples_dir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))+os.sep+"examples"+os.sep+"pysp"+os.sep

pyomo_bin_dir = dirname(dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))))+os.sep+"bin"+os.sep

baseline_dir = this_test_file_directory+"baselines"+os.sep

#
# Import the testing packages
#
import pyutilib.misc
import pyutilib.th as unittest
from pyutilib.pyro import using_pyro3, using_pyro4

from pyomo.pysp.util.misc import (_get_test_nameserver,
                                  _get_test_dispatcher,
                                  _poll,
                                  _kill)
import pyomo.opt
import pyomo.pysp
import pyomo.pysp.phinit
import pyomo.pysp.ef_writer_script
import pyomo.environ

_diff_tolerance = 1e-5
_diff_tolerance_relaxed = 1e-3

def _remove(filename):
    try:
        os.remove(filename)
    except OSError:
        pass

def filter_time_and_data_dirs(line):
    return ("Constructing solvers of type=" in line) or \
           ("Constructing solver type=" in line) or \
           ("Async buffer length=" in line) or \
           ("seconds" in line) or \
           ("starting at" in line) or \
           ("solve ends" in line) or \
           line.startswith("Output file written to") or \
           ("filename" in line) or \
           ("directory" in line) or \
           ("file" in line) or \
           ("module=" in line) or \
           line.startswith("WARNING:") or \
           ("model_location: " in line) or \
           ("model_directory: " in line) or \
           ("scenario_tree_location: " in line) or \
           ("instance_directory: " in line) or \
           line.startswith("Freeing MIP data") or \
           line.startswith("Freeing QP data") or \
           line.startswith("At least one sub-problem solve time was undefined")

def filter_lagrange(line):
    return filter_time_and_data_dirs(line) or \
        ("STARTTIME = ") in line or \
        ("datetime = ") in line or \
        ("lapsed time = ") in line

# pyro output filtering is complex, due to asynchronous behaviors -
# filter all blather regarding what Pyro components are doing.
def filter_pyro(line):
    if line.startswith("URI") or line.startswith("Object URI") or line.startswith("Dispatcher Object URI") or line.startswith("Dispatcher is ready"):
       return True
    elif ("pyro_host: " in line) or ("pyro_port: " in line):
        return True
    elif ("Timeout reached before " in line):
        return True
    elif line.startswith('Client assigned dispatcher with URI'):
        return True
    elif line.startswith("Initializing PH"): # added to prevent diff'ing showing up a positive because of PH initialization order relative to the other pyro-based components
        return True
    elif line.startswith("Applying solver") or line.find("Applying solver") != -1:
       return True
    elif line.startswith("Name server listening on:"):
       return True
    elif line.startswith("Client attempting to find Pyro dispatcher object"):
       return True
    elif line.startswith("Connection to dispatch server established after"):
       return True
    elif line.startswith("Worker attempting to find Pyro dispatcher object"):
       return True
    elif line.startswith("Getting work from"):
       return True
    elif line.startswith("Name Server started."):
       return True
    elif line.startswith("Name Server gracefully stopped."):
       return True
    elif line.startswith("The Name Server appears to be already running on this segment."):
       return True
    elif line.startswith("(host:"):
       return True
    elif line.startswith("Cannot start multiple Name Servers in the same network segment."):
       return True
    elif line.startswith("Listening for work from"):
       return True
    elif line.startswith("Error loading pyomo.opt entry point"): # supressing weird error that occasionally pops up when loading plugins
       return True
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
    elif line.find("Freeing MIP data") != -1: # CPLEX 12.4 python interface idiocy
       return True
    elif line.find("Warning: integer solution") != -1: # CPLEX 12.4 python interface idiocy
       return True
    elif line.startswith("Creating instance"): # non-deterministic order for PH Pyro solver manager
       return True
    elif line.startswith("Exception in thread"): # occasionally raised during Pyro component shutdown
       return True
    elif line.startswith("Traceback"): # occasionally raised during Pyro component shutdown
       return True
    elif line.startswith("File"): # occasionally raised during Pyro component shutdown
       return True
    return filter_time_and_data_dirs(line)

_pyomo_ns_host = '127.0.0.1'
_pyomo_ns_port = None
_pyomo_ns_process = None
_dispatch_srvr_port = None
_dispatch_srvr_process = None
def _setUpModule():
    global _pyomo_ns_port
    global _pyomo_ns_process
    global _dispatch_srvr_port
    global _dispatch_srvr_process
    if _pyomo_ns_process is None:
        _pyomo_ns_process, _pyomo_ns_port = \
            _get_test_nameserver(ns_host=_pyomo_ns_host)
    assert _pyomo_ns_process is not None
    if _dispatch_srvr_process is None:
        _dispatch_srvr_process, _dispatch_srvr_port = \
            _get_test_dispatcher(ns_host=_pyomo_ns_host,
                                 ns_port=_pyomo_ns_port)
    assert _dispatch_srvr_process is not None

def tearDownModule():
    global _pyomo_ns_port
    global _pyomo_ns_process
    global _dispatch_srvr_port
    global _dispatch_srvr_process
    _kill(_pyomo_ns_process)
    _pyomo_ns_port = None
    _pyomo_ns_process = None
    _kill(_dispatch_srvr_process)
    _dispatch_srvr_port = None
    _dispatch_srvr_process = None
    if os.path.exists("gurobi.log"):
        os.remove("gurobi.log")
    if os.path.exists("cplex.log"):
        os.remove("cplex.log")

#
# Define a testing class, using the unittest.TestCase class.
#

solver = {}
solver['cplex','lp'] = False
solver['cplex','python'] = False
solver['gurobi','lp'] = False
solver['gurobi','python'] = False
solver['cbc','lp'] = False
solver['ipopt','nl'] = False

def _setUpClass(cls):
    global solvers
    import pyomo.environ
    from pyomo.solvers.tests.solvers import test_solver_cases
    for _solver, _io in test_solver_cases():
        if (_solver, _io) in solver and \
            test_solver_cases(_solver, _io).available:
            solver[_solver, _io] = True

@unittest.category('nightly', 'performance')
class TestPH(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _setUpClass(cls)

    def setUp(self):
        os.chdir(thisdir)

    def tearDown(self):

        # IMPT: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

    def test_farmer_quadratic_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_cplex.out",
            baseline_dir+"farmer_quadratic_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_nonnormalized_termdiff_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --enable-termdiff-convergence --termdiff-threshold=0.01"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_cplex.out",
            baseline_dir+"farmer_quadratic_nonnormalized_termdiff_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_cplex_direct(self):
        if not solver['cplex','python']:
            self.skipTest("The 'cplex' python solver is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-io=python --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_cplex_direct.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_cplex_direct.out",
            baseline_dir+"farmer_quadratic_cplex_direct.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance_relaxed)

    def test_farmer_quadratic_gurobi_direct(self):
        if not solver['gurobi','python']:
            self.skipTest("The 'gurobi' python solver is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-io=python --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_gurobi_direct.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_gurobi_direct.out",
            baseline_dir+"farmer_quadratic_gurobi_direct.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_gurobi.out",
            baseline_dir+"farmer_quadratic_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_nonnormalized_termdiff_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --enable-termdiff-convergence --termdiff-threshold=0.01"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_gurobi.out",
            baseline_dir+"farmer_quadratic_nonnormalized_termdiff_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_ipopt(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=ipopt --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_ipopt.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_ipopt.out",
            baseline_dir+"farmer_quadratic_ipopt.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_maximize_quadratic_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --verbose --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_maximize_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_maximize_quadratic_gurobi.out",
            baseline_dir+"farmer_maximize_quadratic_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_farmer_with_integers_quadratic_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --default-rho=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_with_integers_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_with_integers_quadratic_cplex.out",
            baseline_dir+"farmer_with_integers_quadratic_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_farmer_with_integers_quadratic_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --default-rho=10"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"farmer_with_integers_quadratic_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
           self.assertFileEqualsBaseline(
               log_output_file,
               baseline_dir+"farmer_with_integers_quadratic_gurobi_darwin.baseline",
               filter=filter_time_and_data_dirs,
               tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"farmer_with_integers_quadratic_gurobi.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"farmer_with_integers_quadratic_gurobi.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_farmer_quadratic_verbose_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_verbose_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_verbose_cplex.out",
            baseline_dir+"farmer_quadratic_verbose_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_verbose_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_verbose_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_verbose_gurobi.out",
            baseline_dir+"farmer_quadratic_verbose_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_trivial_bundling_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_cplex.out",
            baseline_dir+"farmer_quadratic_trivial_bundling_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_trivial_bundling_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_gurobi.out",
            baseline_dir+"farmer_quadratic_trivial_bundling_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_trivial_bundling_ipopt(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "runph --traceback -r 1.0 --solver=ipopt --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt.out",
            baseline_dir+"farmer_quadratic_trivial_bundling_ipopt.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_basic_bundling_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_basic_bundling_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_basic_bundling_cplex.out",
            baseline_dir+"farmer_quadratic_basic_bundling_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_quadratic_basic_bundling_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_quadratic_basic_bundling_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_basic_bundling_gurobi.out",
            baseline_dir+"farmer_quadratic_basic_bundling_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_with_rent_quadratic_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmerWrent"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_with_rent_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_with_rent_quadratic_cplex.out",
            baseline_dir+"farmer_with_rent_quadratic_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_with_rent_quadratic_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmerWrent"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_with_rent_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_with_rent_quadratic_gurobi.out",
            baseline_dir+"farmer_with_rent_quadratic_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        solver_string="cplex"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_linearized_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_cplex.out",
            baseline_dir+"farmer_linearized_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_cbc(self):
        if not solver['cbc','lp']:
            self.skipTest("The 'cbc' executable is not available")
        solver_string="cbc"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_linearized_cbc.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_cbc.out",
            baseline_dir+"farmer_linearized_cbc.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_maximize_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        solver_string="cplex"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_maximize_linearized_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_maximize_linearized_cplex.out",
            baseline_dir+"farmer_maximize_linearized_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        solver_string="gurobi"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_linearized_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_gurobi.out",
            baseline_dir+"farmer_linearized_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_maximize_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        solver_string="gurobi"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_maximize_linearized_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_maximize_linearized_gurobi.out",
            baseline_dir+"farmer_maximize_linearized_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_nodedata_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        solver_string="cplex"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_linearized_nodedata_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_nodedata_cplex.out",
            baseline_dir+"farmer_linearized_nodedata_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_linearized_farmer_nodedata_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        solver_string="gurobi"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph --traceback -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_linearized_nodedata_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_nodedata_gurobi.out",
            baseline_dir+"farmer_linearized_nodedata_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)

    def test_farmer_ef(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_ef.out",
            baseline_dir+"farmer_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"farmer_ef.baseline.lp")

    def test_farmer_maximize_ef(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"farmer_maximize_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" -o max --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_maximize_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_maximize_ef.out",
            baseline_dir+"farmer_maximize_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"farmer_maximize_ef.baseline.lp")

    def test_farmer_piecewise_ef(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWpiecewise"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        ef_output_file = this_test_file_directory+"test_farmer_piecewise_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_piecewise_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_piecewise_ef.out",
            baseline_dir+"farmer_piecewise_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"farmer_piecewise_ef.baseline.lp")

    def test_farmer_ef_with_solve_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_cplex.lp"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file+" --solver=cplex --solve"
        print("Testing command: " + argstring)
        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_ef_with_solve_cplex.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_ef_with_solve_cplex.out",
            baseline_dir+"farmer_ef_with_solve_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    def test_farmer_ef_with_solve_cplex_with_csv_writer(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" --solver=cplex --solve --solution-writer=pyomo.pysp.plugins.csvsolutionwriter"
        print("Testing command: " + argstring)
        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_ef_with_solve_cplex_with_csv_writer.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_ef_with_solve_cplex_with_csv_writer.out",
            baseline_dir+"farmer_ef_with_solve_cplex_with_csv_writer.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        # the following comparison is a bit weird, in that "ef.csv" is written to the current directory.
        # at present, we can't specify a directory for this file in pysp. so, we'll look for it here,
        # and if the test passes, clean up after ourselves if the test passes.
        self.assertFileEqualsBaseline(
            "ef.csv",
            baseline_dir+"farmer_ef_with_solve_cplex_with_csv_writer.csv",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            "ef_StageCostDetail.csv",
            baseline_dir+"farmer_ef_with_solve_cplex_with_csv_writer_StageCostDetail.csv",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)


    def test_farmer_maximize_ef_with_solve_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_maximize_with_solve_cplex.lp"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" -o max --output-file="+ef_output_file+" --solver=cplex --solve"
        print("Testing command: " + argstring)
        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_maximize_ef_with_solve_cplex.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_maximize_ef_with_solve_cplex.out",
            baseline_dir+"farmer_maximize_ef_with_solve_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    def test_farmer_ef_with_solve_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_gurobi.lp"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file+" --solver=gurobi --solve"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_ef_with_solve_gurobi.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_ef_with_solve_gurobi.out",
            baseline_dir+"farmer_ef_with_solve_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    def test_farmer_maximize_ef_with_solve_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_maximize_with_solve_gurobi.lp"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" -o max --output-file="+ef_output_file+" --solver=gurobi --solve"
        print("Testing command: " + argstring)
        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_maximize_ef_with_solve_gurobi.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_maximize_ef_with_solve_gurobi.out",
            baseline_dir+"farmer_maximize_ef_with_solve_gurobi.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    def test_farmer_ef_with_solve_ipopt(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_ipopt.nl"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file+" --solver=ipopt --solve"
        print("Testing command: " + argstring)
        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_ef_with_solve_ipopt.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
           self.assertFileEqualsBaseline(
               this_test_file_directory+"farmer_ef_with_solve_ipopt.out",
               baseline_dir+"farmer_ef_with_solve_ipopt_darwin.baseline",
               filter=filter_time_and_data_dirs,
               tolerance=_diff_tolerance)
        else:
           self.assertFileEqualsBaseline(
               this_test_file_directory+"farmer_ef_with_solve_ipopt.out",
               baseline_dir+"farmer_ef_with_solve_ipopt.baseline",
               filter=filter_time_and_data_dirs,
               tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    def test_hydro_ef(self):
        hydro_examples_dir = pysp_examples_dir + "hydro"
        model_dir = hydro_examples_dir + os.sep + "models"
        instance_dir = hydro_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_hydro_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"hydro_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"hydro_ef.out",
            baseline_dir+"hydro_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"hydro_ef.baseline.lp")

    def test_sizes3_ef(self):
        sizes3_examples_dir = pysp_examples_dir + "sizes"
        model_dir = sizes3_examples_dir + os.sep + "models"
        instance_dir = sizes3_examples_dir + os.sep + "SIZES3"
        ef_output_file = this_test_file_directory+"test_sizes3_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"sizes3_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"sizes3_ef.out",
            baseline_dir+"sizes3_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"sizes3_ef.baseline.lp.gz")

    @unittest.category('fragile')
    def test_sizes3_ef_with_solve_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        sizes3_examples_dir = pysp_examples_dir + "sizes"
        model_dir = sizes3_examples_dir + os.sep + "models"
        instance_dir = sizes3_examples_dir + os.sep + "SIZES3"
        ef_output_file = this_test_file_directory+"test_sizes3_ef.lp"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file+" --solver=cplex --solve"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes3_ef_with_solve_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"sizes3_ef_with_solve_cplex_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_ef_with_solve_cplex.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_ef_with_solve_cplex.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    @unittest.category('fragile')
    def test_sizes3_ef_with_solve_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        sizes3_examples_dir = pysp_examples_dir + "sizes"
        model_dir = sizes3_examples_dir + os.sep + "models"
        instance_dir = sizes3_examples_dir + os.sep + "SIZES3"
        ef_output_file = this_test_file_directory+"test_sizes3_ef.lp"
        argstring = "runef --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file+" --solver=gurobi --solve"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes3_ef_with_solve_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
           self.assertFileEqualsBaseline(
               log_output_file,
               baseline_dir+"sizes3_ef_with_solve_gurobi_darwin.baseline",
               filter=filter_time_and_data_dirs,
               tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_ef_with_solve_gurobi.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_ef_with_solve_gurobi.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)
        self.assertTrue(os.path.exists(ef_output_file))
        os.remove(ef_output_file)

    def test_forestry_ef(self):
        forestry_examples_dir = pysp_examples_dir + "forestry"
        model_dir = forestry_examples_dir + os.sep + "models-nb-yr"
        instance_dir = forestry_examples_dir + os.sep + "18scenarios"
        ef_output_file = this_test_file_directory+"test_forestry_ef.lp"
        argstring = "runef -o max --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"forestry_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"forestry_ef.out",
            baseline_dir+"forestry_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"forestry_ef.baseline.lp.gz",
            tolerance=_diff_tolerance)

    def test_networkflow1ef10_ef(self):
        networkflow1ef10_examples_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow1ef10_examples_dir + os.sep + "models"
        instance_dir = networkflow1ef10_examples_dir + os.sep + "1ef10"
        ef_output_file = this_test_file_directory+"test_networkflow1ef10_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"networkflow1ef10_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"networkflow1ef10_ef.out",
            baseline_dir+"networkflow1ef10_ef.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"networkflow1ef10_ef.baseline.lp.gz")

    def test_farmer_ef_cvar(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_ef_cvar.lp"
        argstring = "runef --symbolic-solver-labels --verbose --generate-weighted-cvar --risk-alpha=0.90 --cvar-weight=0.0 -m "+model_dir+" -s "+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"farmer_ef_cvar.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_ef_cvar.out",
            baseline_dir+"farmer_ef_cvar.baseline.txt",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertFileEqualsBaseline(
            ef_output_file,
            baseline_dir+"farmer_ef_cvar.baseline.lp")

    @unittest.category('fragile')
    def test_cc_ef_networkflow1ef3_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "runef --solver=cplex -m "+model_dir+" -s "+instance_dir+ \
                    " --cc-alpha=0.5" + \
                    " --cc-indicator-var=delta" + \
                    " --solver-options=\"mipgap=0.001\"" + \
                    " --solve"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"cc_ef_networkflow1ef3_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"cc_ef_networkflow1ef3_cplex.baseline-a",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"cc_ef_networkflow1ef3_cplex.baseline-b",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        if (flag_a) and (flag_b):
            print(diffs_a)
            print(diffs_b)
            self.fail("Differences identified relative to all baseline output file alternatives")
        os.remove(log_output_file)

    def test_lagrangian_cc_networkflow1ef3_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "drive_lagrangian_cc.py -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --alpha-min=0.5" + \
                    " --alpha-max=0.5" + \
                    " --ef-solver-options=\"mipgap=0.001\""
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"lagrangian_cc_networkflow1ef3_cplex.out")
        args = argstring.split()
        pyomo.pysp.drive_lagrangian_cc.run(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"lagrangian_cc_networkflow1ef3_cplex.out",
            baseline_dir+"lagrangian_cc_networkflow1ef3_cplex.baseline",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(
            this_test_file_directory+"ScenarioList.csv"))
        os.remove(this_test_file_directory+"ScenarioList.csv")

    @unittest.category('fragile')
    def test_lagrangian_param_1cc_networkflow1ef3_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "lagrangeParam.py -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)
        args = argstring.split()

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"lagrangian_param_1cc_networkflow1ef3_cplex.out")

        import pyomo.pysp.lagrangeParam
        pyomo.pysp.lagrangeParam.run(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"lagrangian_param_1cc_networkflow1ef3_cplex.out",
            baseline_dir+"lagrangian_param_1cc_networkflow1ef3_cplex.baseline",
            filter=filter_lagrange,
            tolerance=_diff_tolerance)
        self.assertTrue(os.path.exists(
            this_test_file_directory+"ScenarioList.csv"))
        os.remove(this_test_file_directory+"ScenarioList.csv")
        self.assertTrue(os.path.exists(
            this_test_file_directory+"OptimalSelections.csv"))
        os.remove(this_test_file_directory+"OptimalSelections.csv")
        self.assertTrue(os.path.exists(
            this_test_file_directory+"PRoptimal.csv"))
        os.remove(this_test_file_directory+"PRoptimal.csv")

    def test_lagrangian_morepr_1cc_networkflow1ef3_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        _remove(baseline_dir+"lagrange_pr_testPRmore.csv")
        argstring = "lagrangeMorePR.py -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+" --csvPrefix="+baseline_dir+"lagrange_pr_test"
        print("Testing command: " + argstring)
        args = argstring.split()

        pyutilib.misc.setup_redirect(
            this_test_file_directory+"lagrangian_morepr_1cc_networkflow1ef3_cplex.out")

        import pyomo.pysp.lagrangeMorePR
        pyomo.pysp.lagrangeMorePR.run(args=args[1:])
        pyutilib.misc.reset_redirect()
        self.assertFileEqualsBaseline(
            this_test_file_directory+"lagrangian_morepr_1cc_networkflow1ef3_cplex.out",
            baseline_dir+"lagrangian_morepr_1cc_networkflow1ef3_cplex.baseline",
            filter=filter_lagrange,
            tolerance=_diff_tolerance)
        _remove(baseline_dir+"lagrange_pr_testPRmore.csv")

@unittest.category('expensive', 'performance')
class TestPHExpensive(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _setUpClass(cls)

    def setUp(self):
        os.chdir(thisdir)

    def tearDown(self):

        # IMPT: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

    def test_computeconf_networkflow1ef10_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        # IMPORTANT - the following code is quite sensitive to the choice of random seed. the particular
        #             seed below yields feasible sub-problem solves when computing the cost of x^ fixed
        #             relative to a new sample size. without this property, you can't compute confidence
        #             intervals (it means you need a bigger sample size).
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        # the random seed is critical for reproducability - computeconf randomly partitions the scenarios into different groups
        argstring = "computeconf --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --fraction-scenarios-for-solve=0.2"+ \
                    " --number-samples-for-confidence-interval=4"+ \
                    " --random-seed=125"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"computeconf_networkflow1ef10_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.computeconf.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"computeconf_networkflow1ef10_cplex_darwin.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"computeconf_networkflow1ef10_cplex_darwin.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"computeconf_networkflow1ef10_cplex.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"computeconf_networkflow1ef10_cplex.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"computeconf_networkflow1ef10_cplex.baseline-c",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b) and (flag_c):
                print(diffs_a)
                print(diffs_b)
                print(diffs_c)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_quadratic_sizes3_cplex(self):
        if (not solver['cplex','lp']) or (not yaml_available):
            self.skipTest("Either the 'cplex' executable is not "
                          "available or PyYAML is not available")
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes3_quadratic_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes3_quadratic_cplex.baseline-a",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes3_quadratic_cplex.baseline-b",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes3_quadratic_cplex.baseline-c",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        if (flag_a) and (flag_b) and (flag_c):
            print(diffs_a)
            print(diffs_b)
            print(diffs_c)
            self.fail("Differences identified relative to all baseline output file alternatives")
        os.remove(log_output_file)

    def test_quadratic_sizes3_cplex_direct(self):
        if (not solver['cplex','python']) or (not yaml_available):
            self.skipTest("The 'cplex' python solver is not "
                          "available or PyYAML is not available")
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-io=python --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes3_quadratic_cplex_direct.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes3_quadratic_cplex_direct.baseline-a",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes3_quadratic_cplex_direct.baseline-b",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes3_quadratic_cplex_direct.baseline-c",
            filter=filter_time_and_data_dirs,
            tolerance=_diff_tolerance)
        if (flag_a) and (flag_b) and (flag_c):
            print(diffs_a)
            print(diffs_b)
            print(diffs_c)
            self.fail("Differences identified relative to all baseline output file alternatives")
        os.remove(log_output_file)

    def test_quadratic_sizes3_gurobi(self):
        if (not solver['gurobi','lp']) or (not yaml_available):
            self.skipTest("Either the 'gurobi' executable is not "
                          "available or PyYAML is not available")

        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes3_quadratic_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi_darwin.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi_darwin.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi.baseline-c",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b) and (flag_c):
                print(diffs_a)
                print(diffs_b)
                print(diffs_c)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_sizes10_quadratic_twobundles_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES10WithTwoBundles"
        argstring = "runph --traceback -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes10_quadratic_twobundles_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes10_quadratic_twobundles_cplex.baseline-a",
            filter=filter_time_and_data_dirs)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            log_output_file,
            baseline_dir+"sizes10_quadratic_twobundles_cplex.baseline-b",
            filter=filter_time_and_data_dirs)
        if (flag_a) and (flag_b):
            print(diffs_a)
            print(diffs_b)
            self.fail("Differences identified relative to all baseline output file alternatives")
        os.remove(log_output_file)

    def test_sizes10_quadratic_twobundles_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES10WithTwoBundles"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sizes10_quadratic_twobundles_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"sizes10_quadratic_twobundles_gurobi_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance_relaxed)
        else:
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"sizes10_quadratic_twobundles_gurobi.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance_relaxed)
    # dlw, Feb 2018; just checking for a stack trace (no baseline)
    def test_sorgw_sizes3_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph --traceback -r 100.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=5 --user-defined-extension=pyomo.pysp.plugins.sorgw"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"sorgw_sizes3_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        os.remove(log_output_file)
        # these files should be here if all is OK
        os.remove(this_test_file_directory+"winterest.ssv")
        os.remove(this_test_file_directory+"wsummary.ssv")
    #
    def test_quadratic_networkflow1ef10_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph --traceback -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=20"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-immediatefixing.cfg"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"networkflow1ef10_quadratic_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"networkflow1ef10_quadratic_cplex_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_quadratic_cplex.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_quadratic_cplex.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_quadratic_networkflow1ef10_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=20"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-immediatefixing.cfg"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"networkflow1ef10_quadratic_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"networkflow1ef10_quadratic_gurobi_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_quadratic_gurobi.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_quadratic_gurobi.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_linearized_networkflow1ef10_cplex(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph --traceback -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"networkflow1ef10_linearized_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_cplex_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_cplex.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_cplex.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_linearized_networkflow1ef10_gurobi(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph --traceback -r 1.0 --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"networkflow1ef10_linearized_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_linearized_forestry_cplex(self):
        if (not solver['cplex','lp']) or (not yaml_available):
            self.skipTest("Either the 'cplex' executable is not "
                          "available or PyYAML is not available")

        forestry_example_dir = pysp_examples_dir + "forestry"
        model_dir = forestry_example_dir + os.sep + "models-nb-yr"
        instance_dir = forestry_example_dir + os.sep + "18scenarios"
        argstring = "runph --traceback -o max --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10" + " --scenario-mipgap=0.05" + " --default-rho=0.001" + \
                    " --rho-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --linearize-nonbinary-penalty-terms=2"+ \
                    " --bounds-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"boundsetter.py" + \
                    " --enable-ww-extension" + " --ww-extension-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph.cfg" + \
                    " --ww-extension-suffixfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph-nb.suffixes" + \
                    " --solve-ef"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"forestry_linearized_cplex.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"forestry_linearized_cplex_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                 baseline_dir+"forestry_linearized_cplex.baseline-a",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"forestry_linearized_cplex.baseline-b",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_linearized_forestry_gurobi(self):
        if (not solver['gurobi','lp']) or (not yaml_available):
            self.skipTest("Either the 'gurobi' executable is not "
                          "available or PyYAML is not available")

        forestry_example_dir = pysp_examples_dir + "forestry"
        model_dir = forestry_example_dir + os.sep + "models-nb-yr"
        instance_dir = forestry_example_dir + os.sep + "18scenarios"
        argstring = "runph --traceback -o max --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10" + " --scenario-mipgap=0.05" + " --default-rho=0.001" + \
                    " --rho-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --linearize-nonbinary-penalty-terms=2"+ \
                    " --bounds-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"boundsetter.py" + \
                    " --enable-ww-extension" + " --ww-extension-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph.cfg" + \
                    " --ww-extension-suffixfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph-nb.suffixes" + \
                    " --solve-ef"
        print("Testing command: " + argstring)
        log_output_file = this_test_file_directory+"forestry_linearized_gurobi.out"
        pyutilib.misc.setup_redirect(log_output_file)
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args[1:])
        pyutilib.misc.reset_redirect()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"forestry_linearized_gurobi_darwin.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)
        else:
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"forestry_linearized_gurobi.baseline",
                filter=filter_time_and_data_dirs,
                tolerance=_diff_tolerance)

@unittest.skipIf(not (using_pyro3 or using_pyro4), "Pyro or Pyro4 is not available")
@unittest.category('parallel','performance')
class TestPHParallel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        _setUpClass(cls)

    def setUp(self):
        os.chdir(thisdir)
        _setUpModule()
        self._taskworker_processes = []

    def tearDown(self):
        try:
            [_poll(proc) for proc in self._taskworker_processes]
        finally:
            [_kill(proc) for proc in self._taskworker_processes]
            self._taskworker_processes = []
        # IMPT: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

    def _setup_pyro_mip_server(self, count):
        assert len(self._taskworker_processes) == 0
        for i in range(count):
            self._taskworker_processes.append(
                subprocess.Popen(["pyro_mip_server", "--traceback"] + \
                                 ["--pyro-host="+str(_pyomo_ns_host)] + \
                                 ["--pyro-port="+str(_pyomo_ns_port)]))

    def _setup_phsolverserver(self, count):
        assert len(self._taskworker_processes) == 0
        for i in range(count):
            self._taskworker_processes.append(
                subprocess.Popen(["phsolverserver", "--traceback"] + \
                                 ["--pyro-host="+str(_pyomo_ns_host)] + \
                                 ["--pyro-port="+str(_pyomo_ns_port)]))

    def test_farmer_quadratic_cplex_with_pyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_cplex_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_cplex_with_pyro.out",
            baseline_dir+"farmer_quadratic_cplex_with_pyro.baseline",
            filter=filter_pyro)

    @unittest.category('fragile')
    def test_farmer_quadratic_cplex_with_phpyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --handshake-with-phpyro --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_cplex_with_phpyro.out",
            baseline_dir+"farmer_quadratic_cplex_with_phpyro.baseline",
            filter=filter_pyro)

    def test_farmer_quadratic_with_bundles_cplex_with_pyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_with_bundles_cplex_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_with_bundles_cplex_with_pyro.out",
            baseline_dir+"farmer_quadratic_with_bundles_cplex_with_pyro.baseline",
            filter=filter_pyro)

    def test_farmer_quadratic_gurobi_with_phpyro(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=gurobi --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_gurobi_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_gurobi_with_phpyro.out",
            baseline_dir+"farmer_quadratic_gurobi_with_phpyro.baseline",
            filter=filter_pyro)

    def test_farmer_linearized_gurobi_with_phpyro(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --linearize-nonbinary-penalty-terms=10 --solver=gurobi --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_linearized_gurobi_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_gurobi_with_phpyro.out",
            baseline_dir+"farmer_linearized_gurobi_with_phpyro.baseline",
            filter=filter_pyro)

    @unittest.category('fragile')
    def test_farmer_quadratic_ipopt_with_pyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=ipopt --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_ipopt_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_ipopt_with_pyro.out",
            baseline_dir+"farmer_quadratic_ipopt_with_pyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_farmer_quadratic_ipopt_with_phpyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=ipopt --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_ipopt_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_ipopt_with_phpyro.out",
            baseline_dir+"farmer_quadratic_ipopt_with_phpyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_farmer_linearized_ipopt_with_phpyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --linearize-nonbinary-penalty-terms=10 --solver=ipopt --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_linearized_ipopt_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_ipopt_with_phpyro.out",
            baseline_dir+"farmer_linearized_ipopt_with_phpyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_farmer_quadratic_trivial_bundling_ipopt_with_phpyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=ipopt --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt_with_phpyro.out",
            baseline_dir+"farmer_quadratic_trivial_bundling_ipopt_with_phpyro.baseline",
            filter=filter_pyro)

    @unittest.category('fragile')
    def test_farmer_quadratic_bundling_ipopt_with_phpyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        self._setup_phsolverserver(2)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=ipopt --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_bundling_ipopt_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_bundling_ipopt_with_phpyro.out",
            baseline_dir+"farmer_quadratic_bundling_ipopt_with_phpyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_quadratic_sizes3_cplex_with_phpyro(self):
        if (not solver['cplex','lp']) or (not yaml_available):
            self.skipTest("The 'cplex' executable is not available "
                          "or PyYAML is not available")

        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        self._setup_phsolverserver(3)
        log_output_file = this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.out"
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=\"mip_tolerances_integrality=1e-7 threads=1\""+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " > "+log_output_file+" 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"sizes3_quadratic_cplex_with_phpyro_darwin.baseline",
                filter=filter_pyro)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_cplex_with_phpyro.baseline-a",
                filter=filter_pyro)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_cplex_with_phpyro.baseline-b",
                filter=filter_pyro)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    @unittest.category('fragile')
    def test_farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --max-iterations=10 --solve-ef --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve.out",
            baseline_dir+"farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_linearized_sizes3_cplex_with_phpyro(self):
        if (not solver['cplex','lp']) or (not yaml_available):
            self.skipTest("The 'cplex' executable is not available "
                          "or PyYAML is not available")

        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        self._setup_phsolverserver(3)
        log_output_file = this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.out"
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --linearize-nonbinary-penalty-terms=4" + \
                    " > "+log_output_file+" 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"sizes3_linearized_cplex_with_phpyro_darwin.baseline",
                filter=filter_pyro)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_linearized_cplex_with_phpyro.baseline-a",
                filter=filter_pyro)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_linearized_cplex_with_phpyro.baseline-b",
                filter=filter_pyro)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_quadratic_sizes3_gurobi_with_phpyro(self):
        if (not solver['gurobi','lp']) or (not yaml_available):
            self.skipTest("The 'gurobi' executable is not available "
                          "or PyYAML is not available")

        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        self._setup_phsolverserver(3)
        log_output_file = this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.out"
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=gurobi --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=\"mip_tolerances_integrality=1e-7 threads=1\""+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " > "+log_output_file+" 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        if os.sys.platform == "darwin":
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi_with_phpyro_darwin.baseline-a",
                filter=filter_pyro,
                tolerance=_diff_tolerance)
            if (flag_a):
                print(diffs_a)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi_with_phpyro.baseline-a",
                filter=filter_pyro,
                tolerance=_diff_tolerance)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"sizes3_quadratic_gurobi_with_phpyro.baseline-b",
                filter=filter_pyro,
                tolerance=_diff_tolerance)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    def test_farmer_ef_with_solve_cplex_with_pyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(1)
        argstring = "runef --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --verbose --solver=cplex --solver-manager=pyro --solve --pyro-shutdown-workers -m "+model_dir+" -s "+instance_dir+" > "+this_test_file_directory+"farmer_ef_with_solve_cplex_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_ef_with_solve_cplex_with_pyro.out",
            baseline_dir+"farmer_ef_with_solve_cplex_with_pyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    def test_farmer_quadratic_async_ipopt_with_pyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=ipopt --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_async_ipopt_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_async_ipopt_with_pyro.out",
            baseline_dir+"farmer_quadratic_async_ipopt_with_pyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    def test_farmer_quadratic_async_gurobi_with_pyro(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=gurobi --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_quadratic_async_gurobi_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_quadratic_async_gurobi_with_pyro.out",
            baseline_dir+"farmer_quadratic_async_gurobi_with_pyro.baseline",
            filter=filter_pyro)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    def test_farmer_linearized_async_gurobi_with_pyro(self):
        if not solver['gurobi','lp']:
            self.skipTest("The 'gurobi' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=gurobi --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10  "+" > "+this_test_file_directory+"farmer_linearized_async_gurobi_with_pyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_async_gurobi_with_pyro.out",
            baseline_dir+"farmer_linearized_async_gurobi_with_pyro.baseline",
            filter=filter_pyro)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    def test_farmer_linearized_async_ipopt_with_pyro(self):
        if not solver['ipopt','nl']:
            self.skipTest("The 'ipopt' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_pyro_mip_server(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=ipopt --solver-manager=pyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10  "+" > "+this_test_file_directory+"farmer_linearized_async_ipopt_with_pyro.out 2>&1"

        print("Testing command: " + argstring)
        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_linearized_async_ipopt_with_pyro.out",
            baseline_dir+"farmer_linearized_async_ipopt_with_pyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    @unittest.category('fragile')
    def test_farmer_with_integers_linearized_cplex_with_phpyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --linearize-nonbinary-penalty-terms=8 --model-directory="+model_dir+" --instance-directory="+instance_dir+" > "+this_test_file_directory+"farmer_with_integers_linearized_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"farmer_with_integers_linearized_cplex_with_phpyro.out",
            baseline_dir+"farmer_with_integers_linearized_cplex_with_phpyro.baseline",
            filter=filter_pyro,
            tolerance=_diff_tolerance)

    # the primary objective of this test is to validate the bare minimum level of functionality on the PH solver server
    # end (solves and rho setting) - obviously should yield the same results as serial PH.
    @unittest.category('fragile')
    def test_simple_quadratic_networkflow1ef10_cplex_with_phpyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        self._setup_phsolverserver(10)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --solver-manager=phpyro --scenario-solver-options=\"threads=1\" --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=5"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " > "+this_test_file_directory+"networkflow1ef10_simple_quadratic_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        self.assertFileEqualsBaseline(
            this_test_file_directory+"networkflow1ef10_simple_quadratic_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_simple_quadratic_cplex_with_phpyro.baseline",
            filter=filter_pyro)

    # builds on the above test, to validate warm-start capabilities; by imposing a migap,
    # executions with and without warm-starts will arrive at different solutions.
    @unittest.category('fragile')
    def test_advanced_quadratic_networkflow1ef10_cplex_with_phpyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        self._setup_phsolverserver(10)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=5"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-mipgaponly.cfg" + \
                    " > "+this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.out",
                baseline_dir+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro_darwin.baseline",
                filter=filter_pyro)
        else:
            self.assertFileEqualsBaseline(
                this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.out",
                baseline_dir+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.baseline",
                filter=filter_pyro)

    def test_linearized_networkflow1ef10_gurobi_with_phpyro(self):
        if (not solver['gurobi','lp']) or (not yaml_available):
            self.skipTest("The 'gurobi' executable is not available "
                          "or PyYAML is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        self._setup_phsolverserver(10)
        log_output_file = this_test_file_directory+"networkflow1ef10_linearized_gurobi_with_phpyro.out"
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=gurobi --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --enable-termdiff-convergence --termdiff-threshold=0.01" + \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-aggressivefixing.cfg"+ \
                    " --ww-extension-suffixfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py" + \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " > "+log_output_file+" 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi_with_phpyro_darwin.baseline",
                filter=filter_pyro)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi_with_phpyro.baseline-a",
                filter=filter_pyro)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi_with_phpyro.baseline-b",
                filter=filter_pyro)
            [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
                log_output_file,
                baseline_dir+"networkflow1ef10_linearized_gurobi_with_phpyro.baseline-c",
                filter=filter_pyro)
            if (flag_a) and (flag_b) and (flag_c):
                print(diffs_a)
                print(diffs_b)
                print(diffs_c)
                self.fail("Differences identified relative to all baseline output file alternatives")
            os.remove(log_output_file)

    @unittest.category('fragile')
    def test_simple_linearized_networkflow1ef3_cplex_with_phpyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef3"
        self._setup_phsolverserver(3)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=4"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py" + \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " > "+this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(
                this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.out",
                baseline_dir+"networkflow1ef3_simple_linearized_cplex_with_phpyro_darwin.baseline",
                filter=filter_pyro)
        else:
            self.assertFileEqualsBaseline(
                this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.out",
                baseline_dir+"networkflow1ef3_simple_linearized_cplex_with_phpyro.baseline",
                filter=filter_pyro)

    @unittest.category('fragile')
    def test_simple_linearized_networkflow1ef10_cplex_with_phpyro(self):
        if not solver['cplex','lp']:
            self.skipTest("The 'cplex' executable is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        self._setup_phsolverserver(10)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py" + \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " > "+this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_simple_linearized_cplex_with_phpyro.baseline-a",
            filter=filter_pyro)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_simple_linearized_cplex_with_phpyro.baseline-b",
            filter=filter_pyro)
        [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_simple_linearized_cplex_with_phpyro.baseline-c",
            filter=filter_pyro)
        if (flag_a) and (flag_b) and (flag_c):
            print(diffs_a)
            print(diffs_b)
            print(diffs_c)
            self.fail("Differences identified relative to all baseline output file alternatives")
        _remove(this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out")

    @unittest.category('fragile')
    def test_advanced_linearized_networkflow1ef10_cplex_with_phpyro(self):
        if (not solver['cplex','lp']) or (not yaml_available):
            self.skipTest("The 'cplex' executable is not available "
                          "or PyYAML is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        self._setup_phsolverserver(10)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --enable-termdiff-convergence --termdiff-threshold=0.01" + \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-aggressivefixing.cfg"+ \
                    " --ww-extension-suffixfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " > "+this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.baseline-a",
            filter=filter_pyro)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.baseline-b",
            filter=filter_pyro)
        [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out",
            baseline_dir+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.baseline-c",
            filter=filter_pyro)
        if (flag_a) and (flag_b) and (flag_c):
            print(diffs_a)
            print(diffs_b)
            print(diffs_c)
            self.fail("Differences identified relative to all baseline output file alternatives")
        _remove(this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out")

    @unittest.category('fragile')
    def test_linearized_networkflow1ef10_cplex_with_bundles_with_phpyro(self):
        if (not solver['cplex','lp']) or (not yaml_available):
            self.skipTest("The 'cplex' executable is not available "
                          "or PyYAML is not available")

        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        self._setup_phsolverserver(5)
        argstring = "runph --pyro-port="+str(_pyomo_ns_port)+" --pyro-host="+str(_pyomo_ns_host)+" --traceback -r 1.0 --solver=cplex --scenario-solver-options=\"threads=1\" --solver-manager=phpyro --shutdown-pyro-workers --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --scenario-bundle-specification="+networkflow_example_dir+os.sep+"10scenario-bundle-specs/FiveBundles.dat" + \
                    " --max-iterations=10"+ \
                    " --enable-termdiff-convergence --termdiff-threshold=0.01" + \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-aggressivefixing.cfg"+ \
                    " --ww-extension-suffixfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " > "+this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out 2>&1"
        print("Testing command: " + argstring)

        _run_cmd(argstring, shell=True)
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out",
            baseline_dir+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.baseline-a",
            filter=filter_pyro)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out",
            baseline_dir+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.baseline-b",
            filter=filter_pyro)
        [flag_c,lineno_c,diffs_c] = pyutilib.misc.compare_file(
            this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out",
            baseline_dir+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.baseline-c",
            filter=filter_pyro)
        if (flag_a) and (flag_b) and (flag_c):
            print(diffs_a)
            print(diffs_b)
            print(diffs_c)
            self.fail("Differences identified relative to all baseline output file alternatives")
        _remove(this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out")

if __name__ == "__main__":
    unittest.main()
