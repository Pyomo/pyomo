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

from pyomo.common.collections import ComponentMap
from pyomo.environ import ConcreteModel, Var, Constraint


class TestComponentMap(unittest.TestCase):
    def test_tuple(self):
        m = ConcreteModel()
        m.v = Var()
        m.c = Constraint(expr=m.v >= 0)
        m.cm = cm = ComponentMap()

        cm[(1, 2)] = 5
        self.assertEqual(len(cm), 1)
        self.assertIn((1, 2), cm)
        self.assertEqual(cm[1, 2], 5)

        cm[(1, 2)] = 50
        self.assertEqual(len(cm), 1)
        self.assertIn((1, 2), cm)
        self.assertEqual(cm[1, 2], 50)

        cm[(1, (2, m.v))] = 10
        self.assertEqual(len(cm), 2)
        self.assertIn((1, (2, m.v)), cm)
        self.assertEqual(cm[1, (2, m.v)], 10)

        cm[(1, (2, m.v))] = 100
        self.assertEqual(len(cm), 2)
        self.assertIn((1, (2, m.v)), cm)
        self.assertEqual(cm[1, (2, m.v)], 100)

        i = m.clone()
        self.assertIn((1, 2), i.cm)
        self.assertIn((1, (2, i.v)), i.cm)
        self.assertNotIn((1, (2, i.v)), m.cm)
        self.assertIn((1, (2, m.v)), m.cm)
        self.assertNotIn((1, (2, m.v)), i.cm)
