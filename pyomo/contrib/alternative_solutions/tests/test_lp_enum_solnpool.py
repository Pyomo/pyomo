#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pe
import pyomo.opt

import pyomo.contrib.alternative_solutions.tests.test_cases as tc
from pyomo.contrib.alternative_solutions import lp_enum
from pyomo.contrib.alternative_solutions import lp_enum_solnpool

from pyomo.common.dependencies import attempt_import

numpy, numpy_available = attempt_import("numpy")
gurobipy, gurobi_available = attempt_import("gurobipy")

#
# TODO: Setup detailed tests here
#


def test_here():
    if numpy_available:
        n = tc.get_pentagonal_pyramid_mip()
        n.x.domain = pe.Reals
        n.y.domain = pe.Reals

        try:
            sols = lp_enum_solnpool.enumerate_linear_solutions_soln_pool(n, tee=True)
        except pyomo.common.errors.ApplicationError as e:
            sols = []

        # TODO - Confirm how solnpools deal with duplicate solutions
        if gurobi_available:
            assert len(sols) == 7
        else:
            assert len(sols) == 0
