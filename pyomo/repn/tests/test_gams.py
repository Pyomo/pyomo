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
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (Block, ConcreteModel, Connector, Constraint,
                           Objective, Var)
from pyomo.repn.plugins.gams_writer import expression_to_string, split_long_line


class GAMSTests(unittest.TestCase):

    def test_power_function_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        self.assertEquals(expression_to_string(
            m.x ** -3, lbl, smap=smap), "power(x1, -3)")
        self.assertEquals(expression_to_string(
            m.x ** 0.33, smap=smap), "x1 ** 0.33")
        self.assertEquals(expression_to_string(
            pow(m.x, 2), smap=smap), "power(x1, 2)")

    def test_fixed_var_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.z.fix(-3)
        lbl = NumericLabeler('x')
        smap = SymbolMap(lbl)
        self.assertEquals(expression_to_string(
            m.x + m.y - m.z, lbl, smap=smap), "x1 + x2 - (-3)")
        m.z.fix(-400)
        self.assertEquals(expression_to_string(
            m.z + m.y - m.z, smap=smap), "(-400) + x2 - (-400)")
        m.z.fix(8.8)
        self.assertEquals(expression_to_string(
            m.x + m.z - m.y, smap=smap), "x1 + (8.8) - x2")
        m.z.fix(-8.8)
        self.assertEquals(expression_to_string(
            m.x * m.z - m.y, smap=smap), "x1*(-8.8) - x2")

    def test_gams_connector_in_active_constraint(self):
        """Test connector in active constraint for GAMS writer."""
        m = ConcreteModel()
        m.b1 = Block()
        m.b2 = Block()
        m.b1.x = Var()
        m.b2.x = Var()
        m.b1.c = Connector()
        m.b1.c.add(m.b1.x)
        m.b2.c = Connector()
        m.b2.c.add(m.b2.x)
        m.c = Constraint(expr=m.b1.c == m.b2.c)
        m.o = Objective(expr=m.b1.x)
        with self.assertRaises(RuntimeError):
            m.write('testgmsfile.gms')

    def test_split_long_line(self):
        pat = "var1 + log(var2 / 9) - "
        line = (pat * 10000) + "x"
        self.assertEqual(split_long_line(line),
            pat * 3478 + "var1 +\nlog(var2 / 9) - " +
            pat * 3477 + "var1 +\nlog(var2 / 9) - " +
            pat * 3043 + "x")


if __name__ == "__main__":
    unittest.main()
