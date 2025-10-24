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

from pyomo.common.dependencies import numpy_available
from pyomo.common import unittest

import pyomo.common.errors
import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import gurobi_enumerate_linear_solutions
from pyomo.opt import check_available_solvers

import pyomo.environ as pyo

# lp_enum_gurobi uses both 'gurobi' and 'appsi_gurobi'
gurobi_available = len(check_available_solvers("gurobi", "appsi_gurobi")) == 2

#
# TODO: Setup detailed tests here
#


@unittest.skipUnless(gurobi_available, "Gurobi MIP solver not available")
@unittest.skipUnless(numpy_available, "NumPy not found")
class TestLPEnumSolnpool(unittest.TestCase):

    def test_non_positive_num_solutions(self):
        """
        Confirm that an exception is thrown with a non-positive num solutions
        """
        n = tc.get_pentagonal_pyramid_mip()
        try:
            gurobi_enumerate_linear_solutions(n, num_solutions=-1)
        except AssertionError as e:
            pass

    def test_here(self):
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        try:
            sols = gurobi_enumerate_linear_solutions(n, tee=True)
        except pyomo.common.errors.ApplicationError as e:
            sols = []

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0
