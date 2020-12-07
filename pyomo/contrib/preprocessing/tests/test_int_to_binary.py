#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Tests the integer to binary variable reformulation."""
import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, Var, Integers, value
from pyomo.environ import TransformationFactory as xfrm
from pyomo.common.log import LoggingIntercept

import logging
from six import StringIO

class TestIntToBinary(unittest.TestCase):
    """Tests integer to binary variable reformulation."""

    def test_int_to_binary(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(0, 5))
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.preprocessing', logging.INFO):
            xfrm('contrib.integer_to_binary').apply_to(m)
        self.assertIn("Reformulating integer variables using the base2 strategy.", output.getvalue())
        reform_blk = m._int_to_binary_reform
        self.assertEqual(len(reform_blk.int_var_set), 1)
        reform_blk.new_binary_var[0, 0].value = 1
        reform_blk.new_binary_var[0, 1].value = 0
        reform_blk.new_binary_var[0, 2].value = 1
        m.x.value = 5
        self.assertEqual(value(reform_blk.integer_to_binary_constraint[0].body), 0)

    def test_int_to_binary_negative(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers, bounds=(-1, 1))
        xfrm('contrib.integer_to_binary').apply_to(m)
        reform_blk = m._int_to_binary_reform
        self.assertEqual(len(reform_blk.int_var_set), 1)
        reform_blk.new_binary_var[0, 0].value = 1
        reform_blk.new_binary_var[0, 1].value = 0
        m.x.value = 0
        # Check that 0 == -1 + 1 * 2^0 + 0 * 2^1
        self.assertEqual(value(reform_blk.integer_to_binary_constraint[0].body), 0)

    def test_integer_var_unbounded(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers)
        with self.assertRaises(ValueError):
            xfrm('contrib.integer_to_binary').apply_to(m)

    def test_no_integer(self):
        m = ConcreteModel()
        m.x = Var()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.preprocessing', logging.INFO):
            xfrm('contrib.integer_to_binary').apply_to(m)

        expected_message = "Model has no free integer variables."
        self.assertIn(expected_message, output.getvalue())


if __name__ == "__main__":
    unittest.main()