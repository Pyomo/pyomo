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

import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Var,
)

class TestCPDebugging(unittest.TestCase):
    def test_debug_infeasibility(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(2, 5))
        m.y = Var(domain=Integers, bounds=(7, 12))
        m.c = Constraint(expr=m.y <= m.x)

        # ESJ TODO: I don't know how to do this without a baseline, which we
        # really don't want...
