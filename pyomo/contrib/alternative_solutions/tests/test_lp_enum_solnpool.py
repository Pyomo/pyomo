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

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import lp_enum
from pyomo.contrib.alternative_solutions import lp_enum_solnpool
from pyomo.opt import check_available_solvers

import pyomo.environ as pyo

# lp_enum_solnpool uses both 'gurobi' and 'appsi_gurobi'
gurobi_available = len(check_available_solvers('gurobi', 'appsi_gurobi')) == 2

#
# TODO: Setup detailed tests here
#


@unittest.skipUnless(gurobi_available, "Gurobi MIP solver not available")
@unittest.skipUnless(numpy_available, "NumPy not found")
class TestLPEnumSolnpool(unittest.TestCase):

    def test_here(self):
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pyo.Reals
        n.y.domain = pyo.Reals

        try:
            sols = lp_enum_solnpool.enumerate_linear_solutions_soln_pool(n, tee=True)
        except pyomo.common.errors.ApplicationError as e:
            sols = []

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0
