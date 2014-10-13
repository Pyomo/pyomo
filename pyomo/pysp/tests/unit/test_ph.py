#
# Get the directory where this script is defined, and where the baseline
# files are located.
#
import os
import sys
from os.path import abspath, dirname

this_test_file_directory = dirname(abspath(__file__))+os.sep

pysp_examples_dir = dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))+os.sep+"examples"+os.sep+"pysp"+os.sep

pyomo_bin_dir = dirname(dirname(dirname(dirname(dirname(dirname(dirname(abspath(__file__))))))))+os.sep+"bin"+os.sep

#
# Import the testing packages
#
import pyutilib.misc
import pyutilib.th as unittest
import pyutilib.subprocess
import pyutilib.services

from pyomo.util.plugin import *
import pyomo.opt
import pyomo.pysp
import pyomo.pysp.phinit
import pyomo.pysp.ef_writer_script
import pyomo.environ

has_yaml = False
try:
    import yaml
    has_yaml = True
except:
    has_yaml = False

solver = pyomo.opt.load_solvers('cplex', '_cplex_direct', 'gurobi', '_gurobi_direct', 'cbc', 'asl:ipopt')

pyutilib.services.register_executable("mpirun")
mpirun_executable = pyutilib.services.registered_executable('mpirun')
mpirun_available = not mpirun_executable is None

def filter_time_and_data_dirs(line):
    return ("seconds" in line) or \
           ("starting at" in line) or \
           ("solve ends" in line) or \
           line.startswith("Output file written to") or \
           ("filename" in line) or \
           ("directory" in line) or \
           ("file" in line) or \
           ("module=" in line) or \
           line.startswith("WARNING:")

def filter_lagrange(line):
    return filter_time_and_data_dirs(line) or \
        ("STARTTIME = ") in line or \
        ("datetime = ") in line or \
        ("lapsed time = ") in line

# pyro output filtering is complex, due to asynchronous behaviors - filter all blather regarding what Pyro components are doing.
def filter_pyro(line):
    if line.startswith("URI") or line.startswith("Object URI") or line.startswith("Dispatcher Object URI") or line.startswith("Dispatcher is ready"):
       return True
    elif line.startswith("Initializing PH"): # added to prevent diff'ing showing up a positive because of PH initialization order relative to the other pyro-based components
        return True
    elif line.startswith("Applying solver"):
       return True
    elif line.startswith("Name server listening on:"):
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


#
# Define a testing class, using the unittest.TestCase class.
#

class TestPH(unittest.TestCase):

    def cleanup(self):

        # IMPT: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_quadratic_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_cplex.out",this_test_file_directory+"farmer_quadratic_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_quadratic_nonnormalized_termdiff_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --enable-termdiff-convergence --termdiff-threshold=0.01"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_cplex.out",this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['_cplex_direct'] is None, "The 'cplex' python solver is not available")
    def test_farmer_quadratic_cplex_direct(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=cplex --solver-io=python --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_cplex_direct.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_cplex_direct.out",this_test_file_directory+"farmer_quadratic_cplex_direct.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['_gurobi_direct'] is None, "The 'gurobi' python solver is not available")
    def test_farmer_quadratic_gurobi_direct(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=gurobi --solver-io=python --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_gurobi_direct.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_gurobi_direct.out",this_test_file_directory+"farmer_quadratic_gurobi_direct.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_quadratic_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_gurobi.out",this_test_file_directory+"farmer_quadratic_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_quadratic_nonnormalized_termdiff_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --enable-termdiff-convergence --termdiff-threshold=0.01"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_gurobi.out",this_test_file_directory+"farmer_quadratic_nonnormalized_termdiff_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_quadratic_gurobi_with_flattening(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --flatten-expressions"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_gurobi_with_flattening.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_gurobi_with_flattening.out",this_test_file_directory+"farmer_quadratic_gurobi_with_flattening.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['asl:ipopt'] is None, "The 'ipopt' executable is not available")
    def test_farmer_quadratic_ipopt(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=ipopt --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_ipopt.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_ipopt.out",this_test_file_directory+"farmer_quadratic_ipopt.baseline", filter=filter_time_and_data_dirs, tolerance=1e-4)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_maximize_quadratic_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --verbose --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_maximize_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_maximize_quadratic_gurobi.out",this_test_file_directory+"farmer_maximize_quadratic_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_with_integers_quadratic_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --default-rho=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_with_integers_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_integers_quadratic_cplex.out",this_test_file_directory+"farmer_with_integers_quadratic_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_with_integers_quadratic_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --default-rho=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_with_integers_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
           self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_integers_quadratic_gurobi.out",this_test_file_directory+"farmer_with_integers_quadratic_gurobi_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        
        else:
           self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_integers_quadratic_gurobi.out",this_test_file_directory+"farmer_with_integers_quadratic_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_quadratic_verbose_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_verbose_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_verbose_cplex.out",this_test_file_directory+"farmer_quadratic_verbose_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_quadratic_verbose_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_verbose_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_verbose_gurobi.out",this_test_file_directory+"farmer_quadratic_verbose_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_quadratic_trivial_bundling_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_trivial_bundling_cplex.out")        
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_trivial_bundling_cplex.out",this_test_file_directory+"farmer_quadratic_trivial_bundling_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        
    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_quadratic_trivial_bundling_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_trivial_bundling_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_trivial_bundling_gurobi.out",this_test_file_directory+"farmer_quadratic_trivial_bundling_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['asl:ipopt'] is None, "The 'ipopt' executable is not available")
    def test_farmer_quadratic_trivial_bundling_ipopt(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "runph -r 1.0 --solver=ipopt --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt.out")        
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt.out",this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt.baseline", filter=filter_time_and_data_dirs, tolerance=1e-4)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_quadratic_basic_bundling_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_basic_bundling_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_basic_bundling_cplex.out",this_test_file_directory+"farmer_quadratic_basic_bundling_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_quadratic_basic_bundling_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_quadratic_basic_bundling_gurobi.out")        
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_basic_bundling_gurobi.out",this_test_file_directory+"farmer_quadratic_basic_bundling_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_with_rent_quadratic_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWrent"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_with_rent_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_rent_quadratic_cplex.out",this_test_file_directory+"farmer_with_rent_quadratic_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_with_rent_quadratic_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWrent"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_with_rent_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_rent_quadratic_gurobi.out",this_test_file_directory+"farmer_with_rent_quadratic_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_linearized_farmer_cplex(self):
        solver_string="cplex"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_linearized_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_cplex.out",this_test_file_directory+"farmer_linearized_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cbc'] is None, "The 'cbc' executable is not available")
    def test_linearized_farmer_cbc(self):
        solver_string="cbc"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_linearized_cbc.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_cbc.out",this_test_file_directory+"farmer_linearized_cbc.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_linearized_farmer_maximize_cplex(self):
        solver_string="cplex"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_maximize_linearized_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_maximize_linearized_cplex.out",this_test_file_directory+"farmer_maximize_linearized_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_linearized_farmer_gurobi(self):
        solver_string="gurobi"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_linearized_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_gurobi.out",this_test_file_directory+"farmer_linearized_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_linearized_farmer_maximize_gurobi(self):
        solver_string="gurobi"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_maximize_linearized_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_maximize_linearized_gurobi.out",this_test_file_directory+"farmer_maximize_linearized_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_linearized_farmer_nodedata_cplex(self):
        solver_string="cplex"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_linearized_nodedata_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_nodedata_cplex.out",this_test_file_directory+"farmer_linearized_nodedata_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_linearized_farmer_nodedata_gurobi(self):
        solver_string="gurobi"
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        argstring = "runph -r 1.0 --solver="+solver_string+" --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_linearized_nodedata_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_nodedata_gurobi.out",this_test_file_directory+"farmer_linearized_nodedata_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None or not has_yaml, "The 'cplex' executable is not available or PyYAML is not available")
    def test_quadratic_sizes3_cplex(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes3_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_cplex.out", this_test_file_directory+"sizes3_quadratic_cplex.baseline-a", filter=filter_time_and_data_dirs, tolerance=1e-5)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_cplex.out", this_test_file_directory+"sizes3_quadratic_cplex.baseline-b", filter=filter_time_and_data_dirs, tolerance=1e-5)
        if (flag_a) and (flag_b):
            print(diffs_a)
            print(diffs_b)
            self.fail("Differences identified relative to all baseline output file alternatives")               

    @unittest.skipIf(solver['_cplex_direct'] is None or not has_yaml, "The 'cplex' python solver is not available or PyYAML is not available")
    def test_quadratic_sizes3_cplex_direct(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph -r 1.0 --solver=cplex --solver-io=python --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes3_quadratic_cplex_direct.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_cplex_direct.out", this_test_file_directory+"sizes3_quadratic_cplex_direct.baseline-a", filter=filter_time_and_data_dirs, tolerance=1e-5)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_cplex_direct.out", this_test_file_directory+"sizes3_quadratic_cplex_direct.baseline-b", filter=filter_time_and_data_dirs, tolerance=1e-5)
        if (flag_a) and (flag_b):
            print(diffs_a)
            print(diffs_b)
            self.fail("Differences identified relative to all baseline output file alternatives")               
        
    @unittest.skipIf(solver['gurobi'] is None or not has_yaml, "The 'gurobi' executable is not available or PyYAML is not available")
    def test_quadratic_sizes3_gurobi(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"
        argstring = "runph -r 1.0 --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes3_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi.out", this_test_file_directory+"sizes3_quadratic_gurobi_darwin.baseline-a", filter=filter_time_and_data_dirs, tolerance=1e-5)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi.out", this_test_file_directory+"sizes3_quadratic_gurobi_darwin.baseline-b", filter=filter_time_and_data_dirs, tolerance=1e-5)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")               
        else:

            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi.out", this_test_file_directory+"sizes3_quadratic_gurobi.baseline-a", filter=filter_time_and_data_dirs, tolerance=1e-5)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi.out", this_test_file_directory+"sizes3_quadratic_gurobi.baseline-b", filter=filter_time_and_data_dirs, tolerance=1e-5)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")               

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_sizes10_quadratic_twobundles_cplex(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES10WithTwoBundles"
        argstring = "runph -r 1.0 --solver=cplex --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"
        print("Testing command: " + argstring)        

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes10_quadratic_twobundles_cplex.out")        
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes10_quadratic_twobundles_cplex.out",this_test_file_directory+"sizes10_quadratic_twobundles_cplex.baseline-a", filter=filter_time_and_data_dirs)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes10_quadratic_twobundles_cplex.out",this_test_file_directory+"sizes10_quadratic_twobundles_cplex.baseline-b", filter=filter_time_and_data_dirs)
        if (flag_a) and (flag_b):
            print(diffs_a)
            print(diffs_b)
            self.fail("Differences identified relative to all baseline output file alternatives")

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_sizes10_quadratic_twobundles_gurobi(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES10WithTwoBundles"
        argstring = "runph -r 1.0 --solver=gurobi --solver-manager=serial --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"
        print("Testing command: " + argstring)                

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes10_quadratic_twobundles_gurobi.out")        
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"sizes10_quadratic_twobundles_gurobi.out",this_test_file_directory+"sizes10_quadratic_twobundles_gurobi_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)                        
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"sizes10_quadratic_twobundles_gurobi.out",this_test_file_directory+"sizes10_quadratic_twobundles_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)                        

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_quadratic_networkflow1ef10_cplex(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=20"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-immediatefixing.cfg"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"networkflow1ef10_quadratic_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_quadratic_cplex.out",this_test_file_directory+"networkflow1ef10_quadratic_cplex_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"networkflow1ef10_quadratic_cplex.out", this_test_file_directory+"networkflow1ef10_quadratic_cplex.baseline-a", filter=filter_time_and_data_dirs, tolerance=1e-5)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"networkflow1ef10_quadratic_cplex.out", this_test_file_directory+"networkflow1ef10_quadratic_cplex.baseline-b", filter=filter_time_and_data_dirs, tolerance=1e-5)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")               

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_quadratic_networkflow1ef10_gurobi(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph -r 1.0 --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=20"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-immediatefixing.cfg"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"networkflow1ef10_quadratic_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_quadratic_gurobi.out",this_test_file_directory+"networkflow1ef10_quadratic_gurobi_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_quadratic_gurobi.out",this_test_file_directory+"networkflow1ef10_quadratic_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_linearized_networkflow1ef10_cplex(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"networkflow1ef10_linearized_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_cplex.out",this_test_file_directory+"networkflow1ef10_linearized_cplex_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_cplex.out",this_test_file_directory+"networkflow1ef10_linearized_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_linearized_networkflow1ef10_gurobi(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "runph -r 1.0 --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"networkflow1ef10_linearized_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_gurobi.out",this_test_file_directory+"networkflow1ef10_linearized_gurobi_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_gurobi.out",this_test_file_directory+"networkflow1ef10_linearized_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None or not has_yaml, "The 'cplex' executable is not available or PyYAML is not available")
    def test_linearized_forestry_cplex(self):
        forestry_example_dir = pysp_examples_dir + "forestry"
        model_dir = forestry_example_dir + os.sep + "models-nb-yr"
        instance_dir = forestry_example_dir + os.sep + "18scenarios"
        argstring = "runph -o max --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10" + " --scenario-mipgap=0.05" + " --default-rho=0.001" + \
                    " --rho-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --linearize-nonbinary-penalty-terms=2"+ \
                    " --bounds-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"boundsetter.py" + \
                    " --enable-ww-extension" + " --ww-extension-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph.cfg" + \
                    " --ww-extension-suffixfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph-nb.suffixes" + \
                    " --solve-ef"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"forestry_linearized_cplex.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"forestry_linearized_cplex.out",this_test_file_directory+"forestry_linearized_cplex_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)            
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"forestry_linearized_cplex.out", this_test_file_directory+"forestry_linearized_cplex.baseline-a", filter=filter_time_and_data_dirs, tolerance=1e-5)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"forestry_linearized_cplex.out", this_test_file_directory+"forestry_linearized_cplex.baseline-b", filter=filter_time_and_data_dirs, tolerance=1e-5)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")               

    @unittest.skipIf(solver['gurobi'] is None or not has_yaml, "The 'gurobi' executable is not available or PyYAML is not available")
    def test_linearized_forestry_gurobi(self):
        forestry_example_dir = pysp_examples_dir + "forestry"
        model_dir = forestry_example_dir + os.sep + "models-nb-yr"
        instance_dir = forestry_example_dir + os.sep + "18scenarios"
        argstring = "runph -o max --solver=gurobi --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10" + " --scenario-mipgap=0.05" + " --default-rho=0.001" + \
                    " --rho-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --linearize-nonbinary-penalty-terms=2"+ \
                    " --bounds-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"boundsetter.py" + \
                    " --enable-ww-extension" + " --ww-extension-cfgfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph.cfg" + \
                    " --ww-extension-suffixfile="+forestry_example_dir+os.sep+"config"+os.sep+"wwph-nb.suffixes" + \
                    " --solve-ef"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"forestry_linearized_gurobi.out")
        args = argstring.split()
        pyomo.pysp.phinit.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"forestry_linearized_gurobi.out",this_test_file_directory+"forestry_linearized_gurobi_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"forestry_linearized_gurobi.out",this_test_file_directory+"forestry_linearized_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    def test_farmer_ef(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef.out",this_test_file_directory+"farmer_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"farmer_ef.baseline.lp")

    def test_farmer_maximize_ef(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"farmer_maximize_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_maximize_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_maximize_ef.out",this_test_file_directory+"farmer_maximize_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"farmer_maximize_ef.baseline.lp")

    def test_farmer_ef_with_flattened_expressions(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_ef_with_flattening.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --flatten-expressions"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef_with_flattening.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_flattening.out",this_test_file_directory+"farmer_ef_with_flattening.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"farmer_ef_with_flattening.baseline.lp")

    def test_farmer_piecewise_ef(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWpiecewise"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "nodedata"
        ef_output_file = this_test_file_directory+"test_farmer_piecewise_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_piecewise_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_piecewise_ef.out",this_test_file_directory+"farmer_piecewise_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"farmer_piecewise_ef.baseline.lp")

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_ef_with_solve_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_cplex.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --solver=cplex --solve"
        print("Testing command: " + argstring)
        
        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef_with_solve_cplex.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_solve_cplex.out",this_test_file_directory+"farmer_ef_with_solve_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_ef_with_solve_cplex_with_csv_writer(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_cplex_with_csv_writer.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --solver=cplex --solve --solution-writer=pyomo.pysp.plugins.csvsolutionwriter"
        print("Testing command: " + argstring)
        
        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef_with_solve_cplex_with_csv_writer.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_solve_cplex_with_csv_writer.out",this_test_file_directory+"farmer_ef_with_solve_cplex_with_csv_writer.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)
        # the following comparison is a bit weird, in that "ef.csv" is written to the current directory.
        # at present, we can't specify a directory for this file in pysp. so, we'll look for it here,
        # and if the test passes, clean up after ourselves if the test passes.
        self.assertFileEqualsBaseline("ef.csv", this_test_file_directory+"farmer_ef_with_solve_cplex_with_csv_writer.csv", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_farmer_maximize_ef_with_solve_cplex(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_maximize_with_solve_cplex.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --output-file="+ef_output_file+" --solver=cplex --solve"
        print("Testing command: " + argstring)
        
        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_maximize_ef_with_solve_cplex.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_maximize_ef_with_solve_cplex.out",this_test_file_directory+"farmer_maximize_ef_with_solve_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_ef_with_solve_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_gurobi.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --solver=gurobi --solve"
        print("Testing command: " + argstring)
        
        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef_with_solve_gurobi.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_solve_gurobi.out",this_test_file_directory+"farmer_ef_with_solve_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_farmer_maximize_ef_with_solve_gurobi(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "maxmodels"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_maximize_with_solve_gurobi.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" -o max --output-file="+ef_output_file+" --solver=gurobi --solve"
        print("Testing command: " + argstring)
        
        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_maximize_ef_with_solve_gurobi.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_maximize_ef_with_solve_gurobi.out",this_test_file_directory+"farmer_maximize_ef_with_solve_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['asl:ipopt'] is None, "The 'ipopt' executable is not available")
    def test_farmer_ef_with_solve_ipopt(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_with_solve_ipopt.nl"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --solver=ipopt --solve"
        print("Testing command: " + argstring)
        
        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef_with_solve_ipopt.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
           self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_solve_ipopt.out",this_test_file_directory+"farmer_ef_with_solve_ipopt_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-4)                
        else:
           self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_solve_ipopt.out",this_test_file_directory+"farmer_ef_with_solve_ipopt.baseline", filter=filter_time_and_data_dirs, tolerance=1e-4)                

    def test_hydro_ef(self):
        hydro_examples_dir = pysp_examples_dir + "hydro"
        model_dir = hydro_examples_dir + os.sep + "models"
        instance_dir = hydro_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_hydro_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"hydro_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"hydro_ef.out",this_test_file_directory+"hydro_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"hydro_ef.baseline.lp")

    def test_sizes3_ef(self):
        sizes3_examples_dir = pysp_examples_dir + "sizes"
        model_dir = sizes3_examples_dir + os.sep + "models"
        instance_dir = sizes3_examples_dir + os.sep + "SIZES3"
        ef_output_file = this_test_file_directory+"test_sizes3_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes3_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"sizes3_ef.out",this_test_file_directory+"sizes3_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"sizes3_ef.baseline.lp.gz")

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_sizes3_ef_with_solve_cplex(self):
        sizes3_examples_dir = pysp_examples_dir + "sizes"
        model_dir = sizes3_examples_dir + os.sep + "models"
        instance_dir = sizes3_examples_dir + os.sep + "SIZES3"
        ef_output_file = this_test_file_directory+"test_sizes3_ef.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --solver=cplex --solve"
        print("Testing command: " + argstring)        

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes3_ef_with_solve_cplex.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"sizes3_ef_with_solve_cplex.out",this_test_file_directory+"sizes3_ef_with_solve_cplex_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)                
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_ef_with_solve_cplex.out",this_test_file_directory+"sizes3_ef_with_solve_cplex.baseline-a", filter=filter_time_and_data_dirs)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_ef_with_solve_cplex.out",this_test_file_directory+"sizes3_ef_with_solve_cplex.baseline-b", filter=filter_time_and_data_dirs)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")


    @unittest.skipIf(solver['gurobi'] is None, "The 'gurobi' executable is not available")
    def test_sizes3_ef_with_solve_gurobi(self):
        sizes3_examples_dir = pysp_examples_dir + "sizes"
        model_dir = sizes3_examples_dir + os.sep + "models"
        instance_dir = sizes3_examples_dir + os.sep + "SIZES3"
        ef_output_file = this_test_file_directory+"test_sizes3_ef.lp"
        argstring = "runef --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file+" --solver=gurobi --solve"
        print("Testing command: " + argstring)        

        pyutilib.misc.setup_redirect(this_test_file_directory+"sizes3_ef_with_solve_gurobi.out")        
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()

        if os.sys.platform == "darwin":
           self.assertFileEqualsBaseline(this_test_file_directory+"sizes3_ef_with_solve_gurobi.out",this_test_file_directory+"sizes3_ef_with_solve_gurobi_darwin.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        
        else:
           self.assertFileEqualsBaseline(this_test_file_directory+"sizes3_ef_with_solve_gurobi.out",this_test_file_directory+"sizes3_ef_with_solve_gurobi.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)        

    def test_forestry_ef(self):
        forestry_examples_dir = pysp_examples_dir + "forestry"
        model_dir = forestry_examples_dir + os.sep + "models-nb-yr"
        instance_dir = forestry_examples_dir + os.sep + "18scenarios"
        ef_output_file = this_test_file_directory+"test_forestry_ef.lp"
        argstring = "runef -o max --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"forestry_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"forestry_ef.out",this_test_file_directory+"forestry_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"forestry_ef.baseline.lp.gz", tolerance=1e-5)

    def test_networkflow1ef10_ef(self):
        networkflow1ef10_examples_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow1ef10_examples_dir + os.sep + "models"
        instance_dir = networkflow1ef10_examples_dir + os.sep + "1ef10"
        ef_output_file = this_test_file_directory+"test_networkflow1ef10_ef.lp"
        argstring = "runef --symbolic-solver-labels --verbose --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"networkflow1ef10_ef.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_ef.out",this_test_file_directory+"networkflow1ef10_ef.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"networkflow1ef10_ef.baseline.lp.gz")

    def test_farmer_ef_cvar(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        ef_output_file = this_test_file_directory+"test_farmer_ef_cvar.lp"
        argstring = "runef --symbolic-solver-labels --verbose --generate-weighted-cvar --risk-alpha=0.90 --cvar-weight=0.0 --model-directory="+model_dir+" --instance-directory="+instance_dir+" --output-file="+ef_output_file
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"farmer_ef_cvar.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_cvar.out",this_test_file_directory+"farmer_ef_cvar.baseline.out", filter=filter_time_and_data_dirs, tolerance=1e-5)
        self.assertFileEqualsBaseline(ef_output_file,this_test_file_directory+"farmer_ef_cvar.baseline.lp")

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_computeconf_networkflow1ef10_cplex(self):
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

        pyutilib.misc.setup_redirect(this_test_file_directory+"computeconf_networkflow1ef10_cplex.out")
        args = argstring.split()
        pyomo.pysp.computeconf.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"computeconf_networkflow1ef10_cplex.out",this_test_file_directory+"computeconf_networkflow1ef10_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_cc_ef_networkflow1ef3_cplex(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "runef --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --cc-alpha=0.5" + \
                    " --cc-indicator-var=delta" + \
                    " solver-options=\"mipgap=0.001\"" + \
                    " --solve"
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"cc_ef_networkflow1ef3_cplex.out")
        args = argstring.split()
        pyomo.pysp.ef_writer_script.main(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"cc_ef_networkflow1ef3_cplex.out",this_test_file_directory+"cc_ef_networkflow1ef3_cplex.baseline-a", filter=filter_time_and_data_dirs)
        [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"cc_ef_networkflow1ef3_cplex.out",this_test_file_directory+"cc_ef_networkflow1ef3_cplex.baseline-b", filter=filter_time_and_data_dirs)
        if (flag_a) and (flag_b):
            print(diffs_a)
            print(diffs_b)
            self.fail("Differences identified relative to all baseline output file alternatives")

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_lagrangian_cc_networkflow1ef3_cplex(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "drive_lagrangian_cc.py -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --alpha-min=0.5" + \
                    " --alpha-max=0.5" + \
                    " --ef-solver-options=\"mipgap=0.001\""
        print("Testing command: " + argstring)

        pyutilib.misc.setup_redirect(this_test_file_directory+"lagrangian_cc_networkflow1ef3_cplex.out")
        args = argstring.split()
        pyomo.pysp.drive_lagrangian_cc.run(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"lagrangian_cc_networkflow1ef3_cplex.out",this_test_file_directory+"lagrangian_cc_networkflow1ef3_cplex.baseline", filter=filter_time_and_data_dirs, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_lagrangian_param_1cc_networkflow1ef3_cplex(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "lagrangeParam.py -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir
        print("Testing command: " + argstring)
        args = argstring.split()

        pyutilib.misc.setup_redirect(this_test_file_directory+"lagrangian_param_1cc_networkflow1ef3_cplex.out")

        import pyomo.pysp.lagrangeParam
        pyomo.pysp.lagrangeParam.run(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"lagrangian_param_1cc_networkflow1ef3_cplex.out",this_test_file_directory+"lagrangian_param_1cc_networkflow1ef3_cplex.baseline", filter=filter_lagrange, tolerance=1e-5)

    @unittest.skipIf(solver['cplex'] is None, "The 'cplex' executable is not available")
    def test_lagrangian_morepr_1cc_networkflow1ef3_cplex(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models-cc"
        instance_dir = networkflow_example_dir + os.sep + "1ef3-cc"
        argstring = "lagrangeMorePR.py -r 1.0 --solver=cplex --model-directory="+model_dir+" --instance-directory="+instance_dir+" --csvPrefix="+this_test_file_directory+"lagrange_pr_test"
        print("Testing command: " + argstring)
        args = argstring.split()

        pyutilib.misc.setup_redirect(this_test_file_directory+"lagrangian_morepr_1cc_networkflow1ef3_cplex.out")

        import pyomo.pysp.lagrangeMorePR
        pyomo.pysp.lagrangeMorePR.run(args=args)
        pyutilib.misc.reset_redirect()
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"lagrangian_morepr_1cc_networkflow1ef3_cplex.out",this_test_file_directory+"lagrangian_morepr_1cc_networkflow1ef3_cplex.baseline", filter=filter_lagrange, tolerance=1e-5)

class TestPHParallel(unittest.TestCase):

    def cleanup(self):

        # IMPT: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_cplex_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=cplex --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_cplex_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_cplex_with_pyro.out",this_test_file_directory+"farmer_quadratic_cplex_with_pyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_cplex_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --handshake-with-phpyro --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_cplex_with_phpyro.out",this_test_file_directory+"farmer_quadratic_cplex_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_with_bundles_cplex_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=cplex --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_with_bundles_cplex_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_with_bundles_cplex_with_pyro.out",this_test_file_directory+"farmer_quadratic_with_bundles_cplex_with_pyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['gurobi'] is None or not mpirun_available, "Either the 'gurobi' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_gurobi_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --solver=gurobi --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_gurobi_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_gurobi_with_phpyro.out",this_test_file_directory+"farmer_quadratic_gurobi_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['gurobi'] is None or not mpirun_available, "Either the 'gurobi' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_linearized_gurobi_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --linearize-nonbinary-penalty-terms=10 --solver=gurobi --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_linearized_gurobi_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_gurobi_with_phpyro.out",this_test_file_directory+"farmer_linearized_gurobi_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_ipopt_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=ipopt --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_ipopt_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_ipopt_with_pyro.out",this_test_file_directory+"farmer_quadratic_ipopt_with_pyro.baseline", filter=filter_pyro, tolerance=1e-4)

    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_ipopt_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --solver=ipopt --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_ipopt_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_ipopt_with_phpyro.out",this_test_file_directory+"farmer_quadratic_ipopt_with_phpyro.baseline", filter=filter_pyro, tolerance=1e-4)

    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_linearized_ipopt_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --linearize-nonbinary-penalty-terms=10 --solver=ipopt --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_linearized_ipopt_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_ipopt_with_phpyro.out",this_test_file_directory+"farmer_linearized_ipopt_with_phpyro.baseline", filter=filter_pyro, tolerance=1e-4)

    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_trivial_bundling_ipopt_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTrivialBundles"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --solver=ipopt --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt_with_phpyro.out",this_test_file_directory+"farmer_quadratic_trivial_bundling_ipopt_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_bundling_ipopt_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodataWithTwoBundles"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 2 phsolverserver : -np 1 runph -r 1.0 --solver=ipopt --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_bundling_ipopt_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_bundling_ipopt_with_phpyro.out",this_test_file_directory+"farmer_quadratic_bundling_ipopt_with_phpyro.baseline", filter=filter_pyro, tolerance=1e-4)

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available or not has_yaml, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available or PyYAML is not available")
    def test_quadratic_sizes3_cplex_with_phpyro(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"        
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : " + \
                    " -np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " >& "+this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)        
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.out",this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro_darwin.baseline", filter=filter_pyro)
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.out", this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.baseline-a", filter=filter_pyro)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.out", this_test_file_directory+"sizes3_quadratic_cplex_with_phpyro.baseline-b", filter=filter_pyro)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")
            
    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --max-iterations=10 --solve-ef --solver=cplex --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve.out",this_test_file_directory+"farmer_with_integers_quadratic_cplex_with_pyro_with_postef_solve.baseline", filter=filter_pyro)                   

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available or not has_yaml, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available or PyYAML is not available")
    def test_linearized_sizes3_cplex_with_phpyro(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"        
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : " + \
                    " -np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --linearize-nonbinary-penalty-terms=4" + \
                    " >& "+this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)        
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.out",this_test_file_directory+"sizes3_linearized_cplex_with_phpyro_darwin.baseline", filter=filter_pyro)        
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.out", this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.baseline-a", filter=filter_pyro)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.out", this_test_file_directory+"sizes3_linearized_cplex_with_phpyro.baseline-b", filter=filter_pyro)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")               
    @unittest.skipIf(solver['gurobi'] is None or not mpirun_available or not has_yaml, "Either the 'gurobi' executable is not available or the 'mpirun' executable is not available or PyYAML is not available")
    def test_quadratic_sizes3_gurobi_with_phpyro(self):
        sizes_example_dir = pysp_examples_dir + "sizes"
        model_dir = sizes_example_dir + os.sep + "models"
        instance_dir = sizes_example_dir + os.sep + "SIZES3"        
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : " + \
                    " -np 1 runph -r 1.0 --solver=gurobi --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=40"+ \
                    " --rho-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"rhosetter.py"+ \
                    " --scenario-solver-options=mip_tolerances_integrality=1e-7"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.cfg"+ \
                    " --ww-extension-suffixfile="+sizes_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " >& "+this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)        
        self.cleanup()

        if os.sys.platform == "darwin":
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.out", this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro_darwin.baseline-a", filter=filter_pyro, tolerance=1e-5)
            # TBD: We should see different baselines here (on darwin)        
            if (flag_a):
                print(diffs_a)
                self.fail("Differences identified relative to all baseline output file alternatives")               
        else:
            [flag_a,lineno_a,diffs_a] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.out", this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.baseline-a", filter=filter_pyro, tolerance=1e-5)
            [flag_b,lineno_b,diffs_b] = pyutilib.misc.compare_file(this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.out", this_test_file_directory+"sizes3_quadratic_gurobi_with_phpyro.baseline-b", filter=filter_pyro, tolerance=1e-5)
            if (flag_a) and (flag_b):
                print(diffs_a)
                print(diffs_b)
                self.fail("Differences identified relative to all baseline output file alternatives")               

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_ef_with_solve_cplex_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runef --verbose --solver=cplex --solver-manager=pyro --solve --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_ef_with_solve_cplex_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_ef_with_solve_cplex_with_pyro.out",this_test_file_directory+"farmer_ef_with_solve_cplex_with_pyro.baseline", filter=filter_pyro)                

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_async_ipopt_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=ipopt --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_async_ipopt_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_async_ipopt_with_pyro.out",this_test_file_directory+"farmer_quadratic_async_ipopt_with_pyro.baseline", filter=filter_pyro, tolerance=1e-4)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    @unittest.skipIf(solver['gurobi'] is None or not mpirun_available, "Either the 'gurobi' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_quadratic_async_gurobi_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=gurobi --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_quadratic_async_gurobi_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_quadratic_async_gurobi_with_pyro.out",this_test_file_directory+"farmer_quadratic_async_gurobi_with_pyro.baseline", filter=filter_pyro)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    @unittest.skipIf(solver['gurobi'] is None or not mpirun_available, "Either the 'gurobi' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_linearized_async_gurobi_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=gurobi --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10  "+" >& "+this_test_file_directory+"farmer_linearized_async_gurobi_with_pyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_async_gurobi_with_pyro.out",this_test_file_directory+"farmer_linearized_async_gurobi_with_pyro.baseline", filter=filter_pyro)

    # async PH with one pyro solver server should yield the same behavior as serial PH.
    @unittest.skipIf(solver['asl:ipopt'] is None or not mpirun_available, "Either the 'ipopt' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_linearized_async_ipopt_with_pyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmer"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 1 pyro_mip_server : -np 1 runph -r 1.0 --solver=ipopt --solver-manager=pyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+" --linearize-nonbinary-penalty-terms=10  "+" >& "+this_test_file_directory+"farmer_linearized_async_ipopt_with_pyro.out"

        print("Testing command: " + argstring)
        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_linearized_async_ipopt_with_pyro.out",this_test_file_directory+"farmer_linearized_async_ipopt_with_pyro.baseline", filter=filter_pyro, tolerance=1e-4)

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_farmer_with_integers_linearized_cplex_with_phpyro(self):
        farmer_examples_dir = pysp_examples_dir + "farmerWintegers"
        model_dir = farmer_examples_dir + os.sep + "models"
        instance_dir = farmer_examples_dir + os.sep + "scenariodata"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : -np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --linearize-nonbinary-penalty-terms=8 --model-directory="+model_dir+" --instance-directory="+instance_dir+" >& "+this_test_file_directory+"farmer_with_integers_linearized_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()
        self.assertFileEqualsBaseline(this_test_file_directory+"farmer_with_integers_linearized_cplex_with_phpyro.out",this_test_file_directory+"farmer_with_integers_linearized_cplex_with_phpyro.baseline", filter=filter_pyro, tolerance=1e-4)        

    # the primary objective of this test is to validate the bare minimum level of functionality on the PH solver server
    # end (solves and rho setting) - obviously should yield the same results as serial PH.
    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")    
    def test_simple_quadratic_networkflow1ef10_cplex_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 10 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=5"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " >& "+this_test_file_directory+"networkflow1ef10_simple_quadratic_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_simple_quadratic_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef10_simple_quadratic_cplex_with_phpyro.baseline", filter=filter_pyro)

    # builds on the above test, to validate warm-start capabilities; by imposing a migap,
    # executions with and without warm-starts will arrive at different solutions.
    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")    
    def test_advanced_quadratic_networkflow1ef10_cplex_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 10 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=5"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-mipgaponly.cfg" + \
                    " >& "+this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro_darwin.baseline", filter=filter_pyro)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef10_advanced_quadratic_cplex_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['gurobi'] is None or not mpirun_available or not has_yaml, "Either the 'gurobi' executable is not available or the 'mpirun' executable is not available or PyYAML is not available")
    def test_linearized_networkflow1ef10_gurobi_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 10 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=gurobi --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --enable-termdiff-convergence --termdiff-threshold=0.01" + \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-aggressivefixing.cfg"+ \
                    " --ww-extension-suffixfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py" + \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " >& "+this_test_file_directory+"networkflow1ef10_linearized_gurobi_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_gurobi_with_phpyro.out",this_test_file_directory+"networkflow1ef10_linearized_gurobi_with_phpyro_darwin.baseline", filter=filter_pyro)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_gurobi_with_phpyro.out",this_test_file_directory+"networkflow1ef10_linearized_gurobi_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_simple_linearized_networkflow1ef3_cplex_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef3"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 3 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=4"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py" + \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " >& "+this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro_darwin.baseline", filter=filter_pyro)   
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef3_simple_linearized_cplex_with_phpyro.baseline", filter=filter_pyro)
            

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available")
    def test_simple_linearized_networkflow1ef10_cplex_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 10 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py" + \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " >& "+this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef10_simple_linearized_cplex_with_phpyro.baseline", filter=filter_pyro)
        
    @unittest.skipIf(solver['cplex'] is None or not mpirun_available or not has_yaml, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available or PyYAML is not available")
    def test_advanced_linearized_networkflow1ef10_cplex_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 10 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
                    " --max-iterations=10"+ \
                    " --enable-termdiff-convergence --termdiff-threshold=0.01" + \
                    " --enable-ww-extensions"+ \
                    " --ww-extension-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph-aggressivefixing.cfg"+ \
                    " --ww-extension-suffixfile="+networkflow_example_dir+os.sep+"config"+os.sep+"wwph.suffixes"+ \
                    " --rho-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"rhosettermixed.py"+ \
                    " --linearize-nonbinary-penalty-terms=8"+ \
                    " --bounds-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"xboundsetter.py"+ \
                    " --aggregate-cfgfile="+networkflow_example_dir+os.sep+"config"+os.sep+"aggregategetter.py"+ \
                    " >& "+this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        if os.sys.platform == "darwin":
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro_darwin.baseline", filter=filter_pyro)
        else:
            self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.out",this_test_file_directory+"networkflow1ef10_advanced_linearized_cplex_with_phpyro.baseline", filter=filter_pyro)

    @unittest.skipIf(solver['cplex'] is None or not mpirun_available or not has_yaml, "Either the 'cplex' executable is not available or the 'mpirun' executable is not available or PyYAML is not available")
    def test_linearized_networkflow1ef10_cplex_with_bundles_with_phpyro(self):
        networkflow_example_dir = pysp_examples_dir + "networkflow"
        model_dir = networkflow_example_dir + os.sep + "models"
        instance_dir = networkflow_example_dir + os.sep + "1ef10"
        argstring = "mpirun -np 1 pyomo_ns : -np 1 dispatch_srvr : -np 5 phsolverserver : " + \
                    "-np 1 runph -r 1.0 --solver=cplex --solver-manager=phpyro --shutdown-pyro --model-directory="+model_dir+" --instance-directory="+instance_dir+ \
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
                    " >& "+this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out"
        print("Testing command: " + argstring)

        os.system(argstring)
        self.cleanup()

        self.assertFileEqualsBaseline(this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.out",this_test_file_directory+"networkflow1ef10_linearized_cplex_with_bundles_with_phpyro.baseline", filter=filter_pyro)                    

TestPH = unittest.category('nightly', 'performance')(TestPH)

TestPHParallel = unittest.category('parallel', 'performance')(TestPHParallel)

if __name__ == "__main__":
    unittest.main()
