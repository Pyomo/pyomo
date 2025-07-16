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

import pyomo.common.unittest as unittest
import pyomo.environ as pyo

from pyomo.contrib.iis.mis import compute_infeasibility_explanation


class TestMIS(unittest.TestCase):
    def test_trivial_quad(self):
        m = pyo.ConcreteModel("Trivial Quad")
        m.x = pyo.Var([1, 2], bounds=(0, 1))
        m.y = pyo.Var(bounds=(0, 1))
        m.c = pyo.Constraint(expr=m.x[1] * m.x[2] == -1)
        m.d = pyo.Constraint(expr=m.x[1] + m.y >= 1)
        # Note: this particular little problem is quadratic
        # As of 18Feb2024 DLW is not sure the explanation code works
        # with solvers other than ipopt
        ipopt = pyo.SolverFactory("ipopt")
        compute_infeasibility_explanation(m, solver=ipopt)
