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

import pyomo.environ as pe
import pyomo.common.unittest as unittest

import pyomo.contrib.alternative_solutions.solnpool as sp
import pyomo.contrib.alternative_solutions.tests.test_cases as tc

class TestSolnPoolUnit(unittest.TestCase):
    
    #TODO: Add test cases.
    '''
    Cases to cover: 
        MIP feasability, 
        MILP feasability, 
        LP feasability (for an LP just one solution should be returned since gurobi cant enumerate over continuous vars)
        For a MIP or MILP we should check that num solutions, rel_opt_gap and abs_opt_gap work
        Pass at least one solver option to make sure that work, e.g. time limit
        
        I have the triagnle problem which should be easy to test with, there is 
        also the knapsack problem. For the LP case we can use the 2d diamond problem
        I don't really have MILP case worked out though, so we may need to create one.
        
        We probably also need a utility to check that a two sets of solutions are the same.
        Maybe this should be an AOS utility since it may be a thing we will want to do often.
    
    '''
    def test_(self):
        m = tc.get_triangle_ip()
        solutions = sp.gurobi_generate_solutions(m, 11)
        

if __name__ == '__main__':
    unittest.main()