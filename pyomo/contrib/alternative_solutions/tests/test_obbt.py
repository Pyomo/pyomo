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
from pyomo.common.collections import ComponentSet
import pyomo.common.unittest as unittest
import pyomo.contrib.alternative_solutions.aos_utils as au

from pyomo.contrib.alternative_solutions.obbt import obbt_analysis
from pyomo.contrib.alternative_solutions.tests.test_cases \
    import get_2d_diamond_problem

class TestOBBTUnit(unittest.TestCase):
    def test_obbt_continuous(self):
        m = get_2d_diamond_problem()
        results, solutions = obbt_analysis(m, solver='cplex')
        self.assertEqual(results.keys(), m.continuous_bounds.keys())
        for var, bounds in results.items():
            assert_array_almost_equal(bounds, m.continuous_bounds[var])
            
    def test_obbt_infeasible(self):
        m = get_2d_diamond_problem()
        m.infeasible_constraint = pe.Constraint(expr=m.x>=10)
        with self.assertRaises(Exception):
            results, solutions = obbt_analysis(m, solver='cplex')

if __name__ == '__main__':
    unittest.main()