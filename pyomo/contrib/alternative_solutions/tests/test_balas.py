#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import Counter

from pyomo.common import unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.dependencies import numpy as numpy, numpy_available
import pyomo.opt

parameterized, param_available = attempt_import('parameterized')
if not param_available:
    raise unittest.SkipTest('Parameterized is not available.')
parameterized = parameterized.parameterized

if numpy_available:
    from numpy.testing import assert_array_almost_equal

from pyomo.contrib.alternative_solutions import enumerate_binary_solutions
import pyomo.contrib.alternative_solutions.tests.test_cases as tc

solvers = list(pyomo.opt.check_available_solvers("glpk", "gurobi", "appsi_gurobi"))


class TestBalasUnit(unittest.TestCase):

    @parameterized.expand(input=solvers)
    def test_bad_solver(self, mip_solver):
        """
        Confirm that an exception is thrown with a bad solver name.
        """
        m = tc.get_triangle_ip()
        with self.assertRaises(pyomo.common.errors.ApplicationError):
            enumerate_binary_solutions(m, solver="unknown_solver")

    @parameterized.expand(input=solvers)
    def test_non_positive_num_solutions(self, mip_solver):
        """
        Confirm that an exception is thrown with a non-positive num solutions
        """
        m = tc.get_triangle_ip()
        with self.assertRaises(AssertionError):
            enumerate_binary_solutions(m, num_solutions=-1, solver=mip_solver)

    @parameterized.expand(input=solvers)
    def test_ip_feasibility(self, mip_solver):
        """
        Enumerate solutions for an ip: triangle_ip.

        Check that there is just one solution when the # of binary variables is 0.
        """
        m = tc.get_triangle_ip()
        results = enumerate_binary_solutions(m, num_solutions=100, solver=mip_solver)
        assert len(results) == 1
        for soln in results:
            assert soln.objective().value == unittest.pytest.approx(5)

    @unittest.skipIf(True, "Ignoring fragile test for solver timeout.")
    @parameterized.expand(input=solvers)
    def test_no_time(self, mip_solver):
        """
        Enumerate solutions for an ip: triangle_ip.

        Check that something sensible happens when the solver times out.
        """
        m = tc.get_triangle_ip()
        with unittest.pytest.raises(Exception):
            results = enumerate_binary_solutions(
                m, num_solutions=100, solver=mip_solver, solver_options={"TimeLimit": 0}
            )

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    @parameterized.expand(input=solvers)
    def test_knapsack_all(self, mip_solver):
        """
        Enumerate solutions for a binary problem: knapsack

        """
        m = tc.get_aos_test_knapsack(
            1, weights=[3, 4, 6, 5], values=[2, 3, 1, 4], capacity=8
        )
        results = enumerate_binary_solutions(m, num_solutions=100, solver=mip_solver)
        objectives = list(
            sorted((round(soln.objective().value, 2) for soln in results), reverse=True)
        )
        assert_array_almost_equal(objectives, m.ranked_solution_values)
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        assert_array_almost_equal(unique_solns_by_obj, m.num_ranked_solns)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    @parameterized.expand(input=solvers)
    def test_knapsack_x0_x1(self, mip_solver):
        """
        Enumerate solutions for a binary problem: knapsack

        Check that we only see 4 solutions that enumerate alternatives of x[1] and x[1]
        """
        m = tc.get_aos_test_knapsack(
            1, weights=[3, 4, 6, 5], values=[2, 3, 1, 4], capacity=8
        )
        results = enumerate_binary_solutions(
            m, num_solutions=100, solver=mip_solver, variables=[m.x[0], m.x[1]]
        )
        objectives = list(
            sorted((round(soln.objective().value, 2) for soln in results), reverse=True)
        )
        assert_array_almost_equal(objectives, [6, 5, 4, 3])
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        assert_array_almost_equal(unique_solns_by_obj, [1, 1, 1, 1])

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    @parameterized.expand(input=solvers)
    def test_knapsack_optimal_3(self, mip_solver):
        """
        Enumerate solutions for a binary problem: knapsack

        """
        m = tc.get_aos_test_knapsack(
            1, weights=[3, 4, 6, 5], values=[2, 3, 1, 4], capacity=8
        )
        results = enumerate_binary_solutions(m, num_solutions=3, solver=mip_solver)
        objectives = list(
            sorted((round(soln.objective().value, 2) for soln in results), reverse=True)
        )
        assert_array_almost_equal(objectives, m.ranked_solution_values[:3])

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    @parameterized.expand(input=solvers)
    def test_knapsack_hamming_3(self, mip_solver):
        """
        Enumerate solutions for a binary problem: knapsack

        """
        m = tc.get_aos_test_knapsack(
            1, weights=[3, 4, 6, 5], values=[2, 3, 1, 4], capacity=8
        )
        results = enumerate_binary_solutions(
            m, num_solutions=3, solver=mip_solver, search_mode="hamming"
        )
        objectives = list(
            sorted((round(soln.objective().value, 2) for soln in results), reverse=True)
        )
        assert_array_almost_equal(objectives, [6, 3, 1])

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    @parameterized.expand(input=solvers)
    def test_knapsack_random_3(self, mip_solver):
        """
        Enumerate solutions for a binary problem: knapsack

        """
        m = tc.get_aos_test_knapsack(
            1, weights=[3, 4, 6, 5], values=[2, 3, 1, 4], capacity=8
        )
        results = enumerate_binary_solutions(
            m, num_solutions=3, solver=mip_solver, search_mode="random", seed=1118798374
        )
        objectives = list(
            sorted((round(soln.objective().value, 2) for soln in results), reverse=True)
        )
        assert_array_almost_equal(objectives, [6, 5, 4])


if __name__ == "__main__":
    unittest.main()
