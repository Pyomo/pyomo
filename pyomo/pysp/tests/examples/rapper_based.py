#  rapper based tests of some PySP examples, started by DLW, Oct 2018
# ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2018 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import sys
import pyutilib.th as unittest
import tempfile
import os
import shutil
import pyomo.environ as pyo
import pyomo.pysp.util.rapper as rapper
import pyomo as pyomoroot

solvername = "ipopt" # could use almost any solver

class Example_via_rapper(unittest.TestCase):
    """ Test some examples using rapper."""

    def setUp(self):
        """pre-test setup"""
        ######## make a temp dir to which files can be copied  #####
        self.tdir = tempfile.mkdtemp()    #TemporaryDirectory().name
        sys.path.insert(1,self.tdir)

        """ During debugging, local files might get in the way
        of finding the file in the temp dir, so we cd there."""
        self.savecwd = os.getcwd()
        os.chdir(self.tdir)

        p = str(pyomoroot.__path__)
        l = p.find("'")
        r = p.find("'", l+1)
        pyomorootpath = p[l+1:r]
        farmpath = pyomorootpath + os.sep + ".." + os.sep + "examples" + \
                   os.sep + "pysp" + os.sep + "farmer"
        self.farmpath = os.path.abspath(farmpath)
        
    def tearDown(self):
        # from GH: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]
            
        os.chdir(self.savecwd)

    def test_famer_netx(self):
        """ solve the ef and check some post solution code"""
        shutil.copyfile(self.farmpath + os.sep + "concreteNetX" +\
                        os.sep + "ReferenceModel.py",
                        self.tdir + os.sep + "ReferenceModel.py")
        import ReferenceModel as RM
        g = RM.pysp_scenario_tree_model_callback()
        stsolver = rapper.StochSolver("ReferenceModel",
                                      fsfct = "pysp_instance_creation_callback",
                                      tree_model  = g)
        ef_sol = stsolver.solve_ef(solvername)
        assert(ef_sol.solver.termination_condition \
               ==  pyo.TerminationCondition.optimal)
        obj = stsolver.root_E_obj()
        assert(abs(-108385 - obj) < 100) # any solver should get this close

if __name__ == '__main__':
    unittest.main()
