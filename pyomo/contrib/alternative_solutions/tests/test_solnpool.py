# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

from collections import Counter

from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common import unittest
from pyomo.contrib.alternative_solutions import gurobi_generate_solutions
from pyomo.contrib.appsi.solvers import Gurobi

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.common.log import LoggingIntercept

gurobipy_available = Gurobi().available()


@unittest.skipIf(not gurobipy_available, "Gurobi MIP solver not available")
class TestSolnPoolUnit(unittest.TestCase):
    """
    Cases to cover:

        LP feasibility (for an LP just one solution should be returned since gurobi cannot enumerate over continuous vars)

        Pass at least one solver option to make sure that work, e.g. time limit

        We need a utility to check that a two sets of solutions are the same.
        Maybe this should be an AOS utility since it may be a thing we will want to do often.
    """

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_ip_feasibility(self):
        """
        Enumerate all solutions for an ip: triangle_ip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_triangle_ip()
        results = gurobi_generate_solutions(m, num_solutions=100)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    def test_ip_num_solutions_best_effort(self):
        """
        Enumerate solutions for an ip: triangle_ip.
        Test best effort mode in solution pool.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_triangle_ip()
        with LoggingIntercept() as LOG:
            results = gurobi_generate_solutions(
                m, num_solutions=8, solver_options={"PoolSearchMode": 1}
            )
        self.assertRegex(
            'Running gurobi_solnpool with PoolSearchMode=1, best effort search may lead to unexpected behavior\n',
            LOG.getvalue(),
        )
        assert len(results) >= 1, 'Need to find some solutions'

    def test_ip_num_solutions_standard_single_solution_solve(self):
        """
        Enumerate solutions for an ip: triangle_ip.
        Test single solve mode in solution pool.

        Check that the correct number of solutions (1) are found.
        This is not the intended use case for this method.
        This is a warning check.
        """
        m = tc.get_triangle_ip()
        with LoggingIntercept() as LOG:
            results = gurobi_generate_solutions(
                m, num_solutions=8, solver_options={"PoolSearchMode": 0}
            )
        self.assertRegex(
            'Running gurobi_solnpool with PoolSearchMode=0, this is single search mode and not the intended use case for gurobi_generate_solutions\n',
            LOG.getvalue(),
        )
        assert len(results) == 1, 'Need to find only 1 solution'

    def test_ip_num_solutions_seeking_one(self):
        """
        Enumerate solutions for an ip: triangle_ip.
        Test case where only one solution is asked for.

        This is not the intended use case for this code.
        This is a warning check.
        """
        m = tc.get_triangle_ip()
        with LoggingIntercept() as LOG:
            results = gurobi_generate_solutions(m, num_solutions=1)
        self.assertRegex(
            'Running alternative_solutions method to find only 1 solution!\n',
            LOG.getvalue(),
        )
        assert len(results) == 1, 'Need to find only 1 solution'

    def test_ip_num_solutions_seeking_zero(self):
        """
        Enumerate solutions for an ip: triangle_ip.
        Test case where zero solutions are asked for to check assert error.
        """
        m = tc.get_triangle_ip()
        with self.assertRaisesRegex(
            AssertionError, "num_solutions must be positive integer"
        ):
            gurobi_generate_solutions(m, num_solutions=0)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_ip_num_solutions(self):
        """
        Enumerate 8 solutions for an ip: triangle_ip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_triangle_ip()
        results = gurobi_generate_solutions(m, num_solutions=8)
        assert len(results) == 8
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = [6, 2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_feasibility(self):
        """
        Enumerate all solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that the correct number of alternate solutions are found.
        """
        m = tc.get_indexed_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(m, num_solutions=100)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_rel_feasibility(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that only solutions within a relative tolerance of 0.2 are
        found.
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(m, num_solutions=100, rel_opt_gap=0.2)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns[0:2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_rel_feasibility_options(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that only solutions within a relative tolerance of 0.2 are
        found.
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(
            m, num_solutions=100, solver_options={"PoolGap": 0.2}
        )
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns[0:2]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_mip_abs_feasibility(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that only solutions within an absolute tolerance of 1.99 are
        found.
        """
        m = tc.get_pentagonal_pyramid_mip()
        results = gurobi_generate_solutions(m, num_solutions=100, abs_opt_gap=1.99)
        objectives = [round(result.objective[1], 2) for result in results]
        actual_solns_by_obj = m.num_ranked_solns[0:3]
        unique_solns_by_obj = [val for val in Counter(objectives).values()]
        np.testing.assert_array_almost_equal(unique_solns_by_obj, actual_solns_by_obj)

    @unittest.skipIf(True, "Ignoring fragile test for solver timeout.")
    def test_mip_no_time(self):
        """
        Enumerate solutions for a mip: indexed_pentagonal_pyramid_mip.

        Check that no solutions are returned with a timelimit of 0.
        """
        m = tc.get_pentagonal_pyramid_mip()
        # Use quiet=False to test error message
        results = gurobi_generate_solutions(
            m, num_solutions=100, solver_options={"TimeLimit": 0.0}, quiet=False
        )
        assert len(results) == 0


if __name__ == "__main__":
    unittest.main()
