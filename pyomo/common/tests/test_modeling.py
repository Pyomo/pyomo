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

from pyomo.environ import ConcreteModel, Var
from pyomo.common.modeling import unique_component_name

class TestModeling(unittest.TestCase):
    def test_unique_component_name(self):
        m = ConcreteModel()
        m.x = 5
        m.y = Var()
        name = unique_component_name(m, 'z')
        self.assertEqual(name, 'z')

        name = unique_component_name(m, 'x')
        self.assertEqual(len(name), 3)
        self.assertEqual(name[:2], 'x_')
        self.assertIn(name[2], '0123456789')

        name = unique_component_name(m, 'y')
        self.assertEqual(len(name), 3)
        self.assertEqual(name[:2], 'y_')
        self.assertIn(name[2], '0123456789')

        name = unique_component_name(m, 'component')
        self.assertEqual(len(name), 11)
        self.assertEqual(name[:10], 'component_')
        self.assertIn(name[10], '0123456789')

        for i in range(10):
            setattr(m, 'y_%s' % i, 0)

        name = unique_component_name(m, 'y')
        self.assertEqual(len(name), 4)
        self.assertEqual(name[:2], 'y_')
        self.assertIn(name[2], '0123456789')
        self.assertIn(name[3], '0123456789')

