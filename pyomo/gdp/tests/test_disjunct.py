#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest

from pyomo.environ import ConcreteModel
from pyomo.gdp import Disjunct, Disjunction

class TestDisjunction(unittest.TestCase):
    def test_empty_disjunction(self):
        m = ConcreteModel()
        m.d = Disjunct()
        m.e = Disjunct()

        m.x1 = Disjunction()
        self.assertEqual(len(m.x1), 0)

        m.x1 = [m.d, m.e]
        self.assertEqual(len(m.x1), 1)
        self.assertEqual(m.x1.disjuncts, [m.d, m.e])

        m.x2 = Disjunction([1,2,3,4])
        self.assertEqual(len(m.x2), 0)

        m.x2[2] = [m.d, m.e]
        self.assertEqual(len(m.x2), 1)
        self.assertEqual(m.x2[2].disjuncts, [m.d, m.e])
