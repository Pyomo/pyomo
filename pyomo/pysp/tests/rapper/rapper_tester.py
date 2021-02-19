#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# Provide some test for rapper; most are smoke because PySP is tested elsewhere
# Author: David L. Woodruff (circa March 2017; Sept 2018; Feb 2020)

import pyutilib.th as unittest
import tempfile
import sys
import os
import shutil
from pyomo.environ import SolverFactory, TerminationCondition
import pyomo.pysp.util.rapper as rapper
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
import pyomo.pysp.plugins.csvsolutionwriter as csvw
import pyomo as pyomoroot
try:
    import networkx
    havenetx = True
except:
    havenetx = False

__author__ = 'David L. Woodruff <DLWoodruff@UCDavis.edu>'
__date__ = 'August 14, 2017'
__version__ = 1.7

solvername = "ipopt" # could use almost any solver
solver_available =  SolverFactory(solvername).available(False)

class Testrapper(unittest.TestCase):
    """ Test the rapper code."""

    def setUp(self):
        """ Get ready for tests"""

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
        farmpath = os.path.abspath(farmpath)
        
        self.farmer_concrete_file = farmpath + os.sep + \
                                    "concrete" + os.sep + "ReferenceModel.py"

        shutil.copyfile(self.farmer_concrete_file,
                        self.tdir + os.sep + "ReferenceModel.py")
        
        abstract_tree = CreateAbstractScenarioTreeModel()
        shutil.copyfile(farmpath + os.sep +"scenariodata" + os.sep + "ScenarioStructure.dat",
                        self.tdir + os.sep + "ScenarioStructure.dat")
        self.farmer_concrete_tree = \
                        abstract_tree.create_instance("ScenarioStructure.dat")
        # added networkx example March 2020
        self.farmer_netx_file = farmpath + os.sep + \
                                    "concreteNetX" + os.sep + "ReferenceModel.py"

        shutil.copyfile(self.farmer_netx_file,
                        self.tdir + os.sep + "NetXReferenceModel.py")
        

    def tearDown(self):
        # from GH: This step is key, as Python keys off the name of the module, not the location.
        #       So, different reference models in different directories won't be detected.
        #       If you don't do this, the symptom is a model that doesn't have the attributes
        #       that the data file expects.
        if "ReferenceModel" in sys.modules:
            del sys.modules["ReferenceModel"]

        sys.path.remove(self.tdir)
        shutil.rmtree(self.tdir, ignore_errors=True)
        os.chdir(self.savecwd)

    def test_fct_contruct(self):
        """ give a callback function rather than a string"""
        from ReferenceModel import pysp_instance_creation_callback
        stsolver = rapper.StochSolver(None,
                                fsfct = pysp_instance_creation_callback,
                                tree_model = self.farmer_concrete_tree)

    def test_no_fsfct_no_tree(self):
        """verify that deprecated concrete with no fsfct is an error"""
        with self.assertRaises(RuntimeError):
            stsolver = rapper.StochSolver("ReferenceModel.py",
                                          fsfct = None,
                                          tree_model = None)

    def test_construct_default_tree_error(self):
        """verify that construction of concrete with default tree name gives error when it should"""
        with self.assertRaises(AttributeError):
            stsolver = rapper.StochSolver("ReferenceModel.py",
                                          fsfct = "pysp_instance_creation_callback",
                                          tree_model = None)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_ef_solve(self):
        """ solve the ef and check some post solution code"""
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                      fsfct = "pysp_instance_creation_callback",
                                tree_model = self.farmer_concrete_tree)
        ef_sol = stsolver.solve_ef(solvername)
        assert(ef_sol.solver.termination_condition \
               ==   TerminationCondition.optimal)
        for name, varval in stsolver.root_Var_solution():
            #print (name, str(varval))
            pass
        self.assertAlmostEqual(varval, 170.0, 1)
        obj = stsolver.root_E_obj()
            
    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_ef_solve_with_gap(self):
        """ solve the ef and report gap"""
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                      fsfct = "pysp_instance_creation_callback",
                                tree_model = self.farmer_concrete_tree)
        res, gap = stsolver.solve_ef(solvername, tee=True, need_gap=True)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_ef_solve_with_csvwriter(self):
        """ solve the ef and report gap"""
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                      fsfct = "pysp_instance_creation_callback",
                                tree_model = self.farmer_concrete_tree)
        res, gap = stsolver.solve_ef(solvername, tee=True, need_gap=True)
        csvw.write_csv_soln(stsolver.scenario_tree, "testcref")
        with open("testcref.csv", 'r') as f:
            line = f.readline()
        assert(line.split(",")[0] == "FirstStage")
            
    def test_ef_cvar_construct(self):
        """ construct the ef with cvar """
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                      fsfct = "pysp_instance_creation_callback",
                                      tree_model = self.farmer_concrete_tree)
        ef = stsolver.make_ef(generate_weighted_cvar = True,
                              cvar_weight = 0.1,
                              risk_alpha = 0.9)

    @unittest.skipIf(not solver_available,
                     "%s solver is not available" % (solvername,))
    def test_ph_solve(self):
        """ use ph"""
        phopts = {'--max-iterations': '2'}
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                      tree_model = self.farmer_concrete_tree,
                                      fsfct = "pysp_instance_creation_callback",
                                      phopts = phopts)
        ph = stsolver.solve_ph(subsolver = solvername, default_rho = 1,
                               phopts=phopts)
        obj = stsolver.root_E_obj() # E[xbar]

        obj, xhat = rapper.xhat_from_ph(ph)

        for nodename, varname, varvalue in rapper.xhat_walker(xhat):
            pass
        assert(nodename == 'RootNode')

    @unittest.skipIf(not solver_available or not havenetx,
                     "solver or NetworkX not available")
    def test_NetX_ef_csvwriter(self):
        """ solve the ef and report gap"""
        import NetXReferenceModel as ref
        tree_model = ref.pysp_scenario_tree_model_callback()
        stsolver = rapper.StochSolver("NetXReferenceModel.py",
                                      fsfct="pysp_instance_creation_callback",
                                      tree_model=tree_model)
        res, gap = stsolver.solve_ef(solvername, tee=True, need_gap=True)
        csvw.write_csv_soln(stsolver.scenario_tree, "testcref")
        with open("testcref_StageCostDetail.csv", 'r') as f:
            line = f.readline()
        assert(line.split(",")[0] == "Stage1")

if __name__ == '__main__':
    unittest.main()
