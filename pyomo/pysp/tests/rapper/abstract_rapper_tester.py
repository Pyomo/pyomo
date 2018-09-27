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

solvername = "gurobi" # could use almost any solver

class Test_abstract_rapper(unittest.TestCase):
    """ Test the rapper code."""

    def setUp(self):
        """ Get ready for tests"""

        p = str(pyomoroot.__path__)
        l = p.find("'")
        r = p.find("'", l+1)
        pyomorootpath = p[l+1:r]
        farmpath = pyomorootpath + os.sep + ".." + os.sep + "examples" + \
                   os.sep + "pysp" + os.sep + "farmer"
        farmpath = os.path.abspath(farmpath)
        
        # for AbstractModels
        self.farmer_ReferencePath = farmpath + os.sep + \
                                    "models" + os.sep + "ReferenceModel.py"
        self.farmer_scenarioPath = farmpath + os.sep + \
                                    "scenariodata"
        
    def tearDown(self):
        pass


    def test_Abstract_Construction(self):
        """ see if we can create the solver object for an AbstractModel"""
             
        stsolver = rapper.StochSolver(self.farmer_ReferencePath,
                                      fsfct = None,
                                      tree_model = self.farmer_scenarioPath,
                                      phopts = None)
        
    def test_Abstract_ef(self):
        """ see if we can create the solver object for an AbstractModel"""
             
        stsolver = rapper.StochSolver(self.farmer_ReferencePath,
                                      fsfct = None,
                                      tree_model = self.farmer_scenarioPath,
                                      phopts = None)
        ef_sol = stsolver.solve_ef(solvername)
        assert(ef_sol.solver.termination_condition \
               ==  pyo.TerminationCondition.optimal)

# see also foo.py        
if __name__ == '__main__':
    unittest.main()
