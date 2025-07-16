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

from pyomo.common.dependencies import numpy as numpy, numpy_available

import pyomo.environ as pyo
from pyomo.common import unittest
import pyomo.opt

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import lp_enum

#
# Find available solvers. Just use GLPK if it's available.
#
solvers = list(
    pyomo.opt.check_available_solvers("glpk", "gurobi")
)  # , "appsi_gurobi"))
pytestmark = unittest.pytest.mark.parametrize("mip_solver", solvers)

timelimit = {"gurobi": "TimeLimit", "appsi_gurobi": "TimeLimit", "glpk": "tmlim"}


@unittest.pytest.mark.default
class TestLPEnum:

    def test_bad_solver(self, mip_solver):
        """
        Confirm that an exception is thrown with a bad solver name.
        """
        m = tc.get_3d_polyhedron_problem()
        try:
            lp_enum.enumerate_linear_solutions(m, solver="unknown_solver")
        except pyomo.common.errors.ApplicationError as e:
            pass

    @unittest.skipIf(True, "Ignoring fragile test for solver timeout.")
    def test_no_time(self, mip_solver):
        """
        Check that the correct bounds are found for a discrete problem where
        more restrictive bounds are implied by the constraints.
        """
        m = tc.get_3d_polyhedron_problem()
        with unittest.pytest.raises(Exception):
            lp_enum.enumerate_linear_solutions(
                m, solver=mip_solver, solver_options={timelimit[mip_solver]: 0}
            )

    def test_3d_polyhedron(self, mip_solver):
        m = tc.get_3d_polyhedron_problem()
        m.o.deactivate()
        m.obj = pyo.Objective(expr=m.x[0] + m.x[1] + m.x[2])

        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        assert len(sols) == 2
        for s in sols:
            assert s.objective_value == unittest.pytest.approx(4)

    def test_3d_polyhedron(self, mip_solver):
        m = tc.get_3d_polyhedron_problem()
        m.o.deactivate()
        m.obj = pyo.Objective(expr=m.x[0] + 2 * m.x[1] + 3 * m.x[2])

        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver)
        assert len(sols) == 2
        for s in sols:
            assert s.objective_value == unittest.pytest.approx(
                9
            ) or s.objective_value == unittest.pytest.approx(10)

    def test_2d_diamond_problem(self, mip_solver):
        m = tc.get_2d_diamond_problem()
        sols = lp_enum.enumerate_linear_solutions(m, solver=mip_solver, num_solutions=2)
        assert len(sols) == 2
        for s in sols:
            print(s)
        assert sols[0].objective_value == unittest.pytest.approx(6.789473684210527)
        assert sols[1].objective_value == unittest.pytest.approx(3.6923076923076916)

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_pentagonal_pyramid(self, mip_solver):
        n = tc.get_pentagonal_pyramid_mip()
        n.o.sense = pyo.minimize
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        sols = lp_enum.enumerate_linear_solutions(n, solver=mip_solver, tee=False)
        for s in sols:
            print(s)
        assert len(sols) == 6

    @unittest.skipIf(not numpy_available, "Numpy not installed")
    def test_pentagon(self, mip_solver):
        n = tc.get_pentagonal_lp()

        sols = lp_enum.enumerate_linear_solutions(n, solver=mip_solver)
        for s in sols:
            print(s)
        assert len(sols) == 6
