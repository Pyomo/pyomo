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
    ConcreteModel,
    Constraint,
    TransformationFactory,
    Var,
)
from pyomo.gdp import Disjunct, Disjunction

class TestCommonConstraintBodyTransformation(unittest.TestCase):
    def create_nested_model(self):
        """
        -100 <= x <= 102
        [-10 <= x <= 11, [x <= 3] v [x >= -17]] v [x == 0]
        """
        m = ConcreteModel()
        m.x = Var(bounds=(-100, 102))
        m.outer_d1 = Disjunct()
        m.outer_d1.c = Constraint(expr=(-10, m.x, 11))
        m.outer_d1.inner_d1 = Disjunct()
        m.outer_d1.inner_d1.c = Constraint(expr=m.x <= 3)
        m.outer_d1.inner_d2 = Disjunct()
        m.outer_d1.inner_d2.c = Constraint(expr=m.x >= -17)
        m.outer_d1.inner = Disjunction(expr=[m.outer_d1.inner_d1,
                                             m.outer_d1.inner_d2])
        m.outer_d2 = Disjunct()
        m.outer_d2.c = Constraint(expr=m.x == 0)
        m.outer = Disjunction(expr=[m.outer_d1, m.outer_d2])

        return m

    def test_transform_nested_model(self):
        m = self.create_nested_model()

        TransformationFactory('gdp.common_constraint_body').apply_to(m)

        # All we need is actually: -17w_2 <= x <= 3w_1

        m.pprint()
        self.assertTrue(False)
