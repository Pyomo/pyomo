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

import pyomo.common.unittest as unittest

from pyomo.environ import (
    TransformationFactory, ConcreteModel, Var, Constraint
)
from pyomo.gdp import Disjunct, Disjunction

class LinearModelDecisionTreeExample(unittest.TestCase):
    def make_model(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(-10, 10))
        m.x2 = Var(bounds=(-20, 20))
        m.d = Var()

        m.d1 = Disjunct()
        m.d1.x1_bounds = Constraint(expr=(0, m.x1, 2))
        m.d1.x2_bounds = Constraint(expr=(0, m.x2, 3))
        m.d1.func = Constraint(expr=m.x1 + m.x2 == m.d)

        m.d2 = Disjunct()
        m.d2.x1_bounds = Constraint(expr=(0, m.x1, 2))
        m.d2.x2_bounds = Constraint(expr=(3, m.x2, 10))
        m.d2.func = Constraint(expr=2*m.x1 + 4*m.x2 + 7 == m.d)

        m.d3 = Disjunct()
        m.d3.x1_bounds = Constraint(expr=(2, m.x1, 10))
        m.d3.x2_bounds = Constraint(expr=(0, m.x2, 1))
        m.d3.func = Constraint(expr=m.x1 - 5*m.x2 - 3 == m.d)

        m.disjunction = Disjunction(expr=[m.d1, m.d2, m.d3])
        
        return m

    def test_calculated_Ms_correct(self):
        m = self.make_model()

        TransformationFactory('gdp.mbigm').apply_to(m)
