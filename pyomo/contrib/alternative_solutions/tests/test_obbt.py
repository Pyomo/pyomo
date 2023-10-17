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

from numpy.testing import assert_array_almost_equal

import pyomo.environ as pe
import pyomo.common.unittest as unittest

from pyomo.contrib.alternative_solutions.obbt import obbt_analysis
import pyomo.contrib.alternative_solutions.tests.test_cases as tc
import pdb

mip_solver = 'gurobi'

class TestOBBTUnit(unittest.TestCase):
    
    #TODO: Add more test cases
    '''
    So far I have added test cases for the feasibility problems, we should test cases
    where we put TODO: objective constraints in as well based on the absolute and relative difference.
    
    Add a case where bounds are only found for a subset of variables.
    
    Try cases where refine_discrete_bounds is set to true to ensure that new constraints are
    added to refine the bounds. I created the problem get_implied_bound_ip to facilitate this
    
    TODO: Check to see that warm starting works for a MIP and MILP case
    
    TODO: We should also check that warmstarting and refining bounds works for gurobi and appsi_gurobi
    
    We should pass at least one solver_options to ensure this work (e.g. time limit)
    
    I only looked at linear cases here, so you think others are worth testing, some simple non-linear (convex) cases?
    
    '''
    
    def obbt_continuous(self):
        '''Check that the correct bounds are found for a continuous problem.'''
        m = tc.get_2d_diamond_problem()
        results = obbt_analysis(m, solver=mip_solver)
        self.assertEqual(results.keys(), m.continuous_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def test_obbt_mip(self):
        '''Check that bound tightening only occurs for a subset of variables.'''
        m = tc.get_bloated_hexagonal_pyramid_mip()
        m.x = 0
        m.y = 0
        m.z = 5
        results = obbt_analysis(m, solver=mip_solver, tee = True, warmstart = True)
        bounds_tightened = False
        bounds_not_tightned = False
        for var, bounds in results.items():
            if bounds[0] > var.lb:
                bounds_tightened = True
            else:
                bounds_not_tightened = True
            if bounds[1] < var.ub:
                bounds_tightened = True
            else:
                bounds_not_tightened = True
        self.assertTrue(bounds_tightened)
        self.assertTrue(bounds_not_tightened)

    def obbt_unbounded(self):
        '''Check that the correct bounds are found for an unbounded problem.'''
        m = tc.get_2d_unbounded_problem()
        results = obbt_analysis(m, solver=mip_solver)
        self.assertEqual(results.keys(), m.continuous_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])

    def bound_tightening(self):
        '''
        Check that the correct bounds are found for a discrete problem where 
        more restrictive bounds are implied by the constraints.'''
        m = tc.get_implied_bound_ip()
        results = obbt_analysis(m, solver=mip_solver)
        self.assertEqual(results.keys(), m.var_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.var_bounds[var])

    def bound_refinement(self):
        '''
        Check that the correct bounds are found for a discrete problem where 
        more restrictive bounds are implied by the constraints.'''
        m = tc.get_implied_bound_ip()
        results = obbt_analysis(m, solver=mip_solver, refine_discrete_bounds=True)
        for var, bounds in results.items():
            if m.var_bounds[var][0] > var.lb:
                self.assertTrue(hasattr(m._obbt, var.name + "_lb"))
            if m.var_bounds[var][1] < var.ub:
                self.assertTrue(hasattr(m._obbt, var.name + "_ub"))
            
    def obbt_infeasible(self):
        '''Check that code catches cases where the problem is infeasible.'''
        m = tc.get_2d_diamond_problem()
        m.infeasible_constraint = pe.Constraint(expr=m.x>=10)
        with self.assertRaises(Exception):
            obbt_analysis(m, solver=mip_solver)

if __name__ == '__main__':
    unittest.main()