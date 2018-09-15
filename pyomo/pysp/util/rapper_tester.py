# Provide some test for rapper
# Author: David L. Woodruff (circa March 2017 and Sept 2018)

import unittest
import tempfile
import sys
import os
import shutil
import json
import pyomo.environ as pyo
import pyomo.pysp.util.rapper as rapper
from pyomo.pysp.scenariotree.tree_structure_model import CreateAbstractScenarioTreeModel
import pyomo as pyomoroot

__author__ = 'David L. Woodruff <DLWoodruff@UCDavis.edu>'
__date__ = 'August 14, 2017'
__version__ = 1.4

solvername = "cplex" # could use almost any solver

class Testrapper(unittest.TestCase):
    """ Test the rapper code."""

    def setUp(self):
        """ Get ready for tests"""

        ######## make a temp dir to which files can be copied  #####
        self.tdir = tempfile.mkdtemp()    #TemporaryDirectory().name
        sys.path.insert(1,self.tdir)

        """ During debugging, local files might get in the way
        of finding the file in the temp dir, so we cd there."""
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
        
    def tearDown(self):
        pass

    def test_ef_solve(self):
        """ solve the ef and check some post solution code"""
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                tree_model = self.farmer_concrete_tree)
        ef_sol = stsolver.solve_ef(solvername)
        assert(ef_sol.solver.termination_condition \
               ==  pyo.TerminationCondition.optimal)
        for name, varval in stsolver.root_Var_solution():
            #print (name, str(varval))
            pass
        self.assertAlmostEqual(varval, 170.0, 1)
        obj = stsolver.root_E_obj()
            
    def test_ef_solve_with_gap(self):
        """ solve the ef and report gap"""
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                tree_model = self.farmer_concrete_tree)
        res, gap = stsolver.solve_ef(solvername, tee=True, need_gap=True)

    def test_ph_solve(self):
        """ use ph; assumes concrete two-stage json passes"""
        phopts = {'--max-iterations': '2'}
        stsolver = rapper.StochSolver("ReferenceModel.py",
                                      tree_model = self.farmer_concrete_tree,
                                      phopts = phopts)
        ph = stsolver.solve_ph(subsolver = solvername, default_rho = 1,
                               phopts=phopts)
        obj = stsolver.root_E_obj() # E[xbar]

        obj, xhat = rapper.xhat_from_ph(ph)

        for nodename, varname, varvalue in rapper.xhat_walker(xhat):
            pass
        assert(nodename == 'RootNode')

if __name__ == '__main__':
    unittest.main()
