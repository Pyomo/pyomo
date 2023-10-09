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

import pyomo.contrib.alternative_solutions.balas
import pyomo.contrib.alternative_solutions.tests.test_cases as tc

class TestBalasUnit(unittest.TestCase):
    
    #TODO: Add test cases
    '''
    Repeat a lot of the test from solnpool to check that the various arguments work correct.
    The main difference will be that we will only want to check binary problems here.
    The knapsack problem should be useful (just set the bounds to 0-1).
    
    The only other thing to test is the different search modes. They should still enumerate 
    all of the solutions, just in a different sequence.
    
    '''
    
    def test_(self):
        pass

if __name__ == '__main__':
    unittest.main()