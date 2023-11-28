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
from pyomo.contrib.mindtpy.util import set_var_value

from pyomo.environ import Var, Integers, ConcreteModel, Integers


class UnitTestMindtPy(unittest.TestCase):
    def test_set_var_value(self):
        m = ConcreteModel()
        m.x1 = Var(within=Integers, bounds=(-1, 4), initialize=0)

        set_var_value(
            m.x1,
            var_val=5,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, 4)

        set_var_value(
            m.x1,
            var_val=-2,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, -1)

        set_var_value(
            m.x1,
            var_val=1.1,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=True,
        )
        self.assertEqual(m.x1.value, 1.1)

        set_var_value(
            m.x1,
            var_val=2.00000001,
            integer_tolerance=1e-6,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, 2)

        set_var_value(
            m.x1,
            var_val=0.0000001,
            integer_tolerance=1e-9,
            zero_tolerance=1e-6,
            ignore_integrality=False,
        )
        self.assertEqual(m.x1.value, 0)


if __name__ == '__main__':
    unittest.main()
