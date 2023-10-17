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

import unittest
from numpy.testing import assert_array_almost_equal

import pyomo.environ as pe
import pyomo.common.unittest as unittest

import pyomo.contrib.alternative_solutions.solnpool as sp
import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from collections import Counter
import pdb

mip_solver = 'gurobi'

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

    def test_ip_feasibility(self):
        '''
        COMMENTS'''
        m = tc.get_triangle_ip()
        results = sp.gurobi_generate_solutions(m, 100)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    def test_mip_feasibility(self):
        '''
        COMMENTS'''
        m = tc.get_hexagonal_pyramid_mip()
        results = sp.gurobi_generate_solutions(m, 100)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    def test_mip_rel_feasibility(self):
        '''
        COMMENTS'''
        m = tc.get_hexagonal_pyramid_mip()
        results = sp.gurobi_generate_solutions(m, 100, rel_opt_gap=.2)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns[0:2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    def test_mip_abs_feasibility(self):
        '''
        COMMENTS'''
        m = tc.get_hexagonal_pyramid_mip()
        results = sp.gurobi_generate_solutions(m, 100, abs_opt_gap=1.99)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns[0:3]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

if __name__ == '__main__':
    unittest.main()
