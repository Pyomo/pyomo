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
from pyomo.contrib.cp.interval_var import (
    IntervalVar,
    IntervalVarTimePoint,
    IntervalVarLength,
    IntervalVarPresence,
)
from pyomo.core.expr import GetItemExpression, GetAttrExpression
from pyomo.environ import ConcreteModel, Integers, Set, value, Var


class TestScalarIntervalVar(unittest.TestCase):
    def test_initialize_with_no_data(self):
        m = ConcreteModel()
        m.i = IntervalVar()

        self.assertIsInstance(m.i.start_time, IntervalVarTimePoint)
        self.assertEqual(m.i.start_time.domain, Integers)
        self.assertIsNone(m.i.start_time.lower)
        self.assertIsNone(m.i.start_time.upper)

        self.assertIsInstance(m.i.end_time, IntervalVarTimePoint)
        self.assertEqual(m.i.end_time.domain, Integers)
        self.assertIsNone(m.i.end_time.lower)
        self.assertIsNone(m.i.end_time.upper)

        self.assertIsInstance(m.i.length, IntervalVarLength)
        self.assertEqual(m.i.length.domain, Integers)
        self.assertIsNone(m.i.length.lower)
        self.assertIsNone(m.i.length.upper)

        self.assertIsInstance(m.i.is_present, IntervalVarPresence)

    def test_add_components_that_do_not_belong(self):
        m = ConcreteModel()
        m.i = IntervalVar()

        with self.assertRaisesRegex(
            ValueError,
            "Attempting to declare a block component using the name of a "
            "reserved attribute:\n\tnew_thing",
        ):
            m.i.new_thing = IntervalVar()

    def test_start_and_end_bounds(self):
        m = ConcreteModel()
        m.i = IntervalVar(start=(0, 5))
        self.assertEqual(m.i.start_time.lower, 0)
        self.assertEqual(m.i.start_time.upper, 5)

        m.i.end_time.bounds = (12, 14)

        self.assertEqual(m.i.end_time.lower, 12)
        self.assertEqual(m.i.end_time.upper, 14)

    def test_constant_length_and_start(self):
        m = ConcreteModel()
        m.i = IntervalVar(length=7, start=3)

        self.assertEqual(m.i.length.lower, 7)
        self.assertEqual(m.i.length.upper, 7)

        self.assertEqual(m.i.start_time.lower, 3)
        self.assertEqual(m.i.start_time.upper, 3)

    def test_non_optional(self):
        m = ConcreteModel()
        m.i = IntervalVar(length=2, end=(4, 9), optional=False)

        self.assertEqual(value(m.i.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i.optional)

        # Should also be true by default
        m.i2 = IntervalVar()

        self.assertEqual(value(m.i2.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i2.optional)

    def test_optional(self):
        m = ConcreteModel()
        m.i = IntervalVar(optional=True)

        self.assertFalse(m.i.is_present.fixed)
        self.assertTrue(m.i.optional)

        # Now set to False
        m.i.optional = False
        self.assertEqual(value(m.i.is_present), True)
        self.assertTrue(m.i.is_present.fixed)
        self.assertFalse(m.i.optional)

    def test_is_present_fixed_False(self):
        m = ConcreteModel()
        m.i = IntervalVar(optional=True)

        m.i.is_present.fix(False)
        self.assertTrue(m.i.optional)


class TestIndexedIntervalVar(unittest.TestCase):
    def test_initialize_with_no_data(self):
        m = ConcreteModel()

        m.i = IntervalVar([1, 2])

        for j in [1, 2]:
            self.assertIsInstance(m.i[j].start_time, IntervalVarTimePoint)
            self.assertEqual(m.i[j].start_time.domain, Integers)
            self.assertIsNone(m.i[j].start_time.lower)
            self.assertIsNone(m.i[j].start_time.upper)

            self.assertIsInstance(m.i[j].end_time, IntervalVarTimePoint)
            self.assertEqual(m.i[j].end_time.domain, Integers)
            self.assertIsNone(m.i[j].end_time.lower)
            self.assertIsNone(m.i[j].end_time.upper)

            self.assertIsInstance(m.i[j].length, IntervalVarLength)
            self.assertEqual(m.i[j].length.domain, Integers)
            self.assertIsNone(m.i[j].length.lower)
            self.assertIsNone(m.i[j].length.upper)

            self.assertIsInstance(m.i[j].is_present, IntervalVarPresence)

    def test_constant_length(self):
        m = ConcreteModel()
        m.i = IntervalVar(['a', 'b'], length=45)

        for j in ['a', 'b']:
            self.assertEqual(m.i[j].length.lower, 45)
            self.assertEqual(m.i[j].length.upper, 45)

    def test_rule_based_start(self):
        m = ConcreteModel()

        def start_rule(m, i):
            return (1 - i, 13 + i)

        m.act = IntervalVar([1, 2, 3], start=start_rule, length=4)

        for i in [1, 2, 3]:
            self.assertEqual(m.act[i].start_time.lower, 1 - i)
            self.assertEqual(m.act[i].start_time.upper, 13 + i)

            self.assertEqual(m.act[i].length.lower, 4)
            self.assertEqual(m.act[i].length.upper, 4)

            self.assertFalse(m.act[i].optional)
            self.assertTrue(m.act[i].is_present.fixed)
            self.assertEqual(value(m.act[i].is_present), True)

    def test_optional(self):
        m = ConcreteModel()
        m.act = IntervalVar([1, 2], end=[0, 10], optional=True)

        for i in [1, 2]:
            self.assertTrue(m.act[i].optional)
            self.assertFalse(m.act[i].is_present.fixed)

            self.assertEqual(m.act[i].end_time.lower, 0)
            self.assertEqual(m.act[i].end_time.upper, 10)

        # None doesn't make sense for this:
        with self.assertRaisesRegex(
            ValueError, "Cannot set 'optional' to None: Must be True or False."
        ):
            m.act[1].optional = None

        # We can change it, and that has the correct effect on is_present
        m.act[1].optional = False
        self.assertFalse(m.act[1].optional)
        self.assertTrue(m.act[1].is_present.fixed)

        m.act[1].optional = True
        self.assertTrue(m.act[1].optional)
        self.assertFalse(m.act[1].is_present.fixed)

    def test_optional_rule(self):
        m = ConcreteModel()
        m.idx = Set(initialize=[(4, 2), (5, 2)], dimen=2)

        def optional_rule(m, i, j):
            return i % j == 0

        m.act = IntervalVar(m.idx, optional=optional_rule)
        self.assertTrue(m.act[4, 2].optional)
        self.assertFalse(m.act[5, 2].optional)

    def test_index_by_expr(self):
        m = ConcreteModel()
        m.act = IntervalVar([(1, 2), (2, 1), (2, 2)])
        m.i = Var(domain=Integers)
        m.i2 = Var([1, 2], domain=Integers)

        thing1 = m.act[m.i, 2]
        self.assertIsInstance(thing1, GetItemExpression)
        self.assertEqual(len(thing1.args), 3)
        self.assertIs(thing1.args[0], m.act)
        self.assertIs(thing1.args[1], m.i)
        self.assertEqual(thing1.args[2], 2)

        thing2 = thing1.start_time
        self.assertIsInstance(thing2, GetAttrExpression)
        self.assertEqual(len(thing2.args), 2)
        self.assertIs(thing2.args[0], thing1)
        self.assertEqual(thing2.args[1], 'start_time')

        # TODO: But this is where it dies.
        expr1 = m.act[m.i, 2].start_time.before(m.act[m.i**2, 1].end_time)
