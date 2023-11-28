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

import logging
import math

import pyomo.common.unittest as unittest
from io import StringIO

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import DeveloperError, InvalidValueError
from pyomo.common.log import LoggingIntercept
from pyomo.core.expr import (
    ProductExpression,
    NPV_ProductExpression,
    SumExpression,
    DivisionExpression,
    NPV_DivisionExpression,
)
from pyomo.environ import (
    ConcreteModel,
    Block,
    Constraint,
    Var,
    Param,
    Objective,
    Suffix,
    Expression,
    Set,
    SortComponents,
)
import pyomo.repn.util
from pyomo.repn.util import (
    _CONSTANT,
    BeforeChildDispatcher,
    ExitNodeDispatcher,
    FileDeterminism,
    FileDeterminism_to_SortComponents,
    InvalidNumber,
    apply_node_operation,
    categorize_valid_components,
    complex_number_error,
    ftoa,
    initialize_var_map_from_column_order,
    ordered_active_constraints,
)

try:
    import numpy as np

    numpy_available = True
except:
    numpy_available = False


class TestRepnUtils(unittest.TestCase):
    def test_ftoa(self):
        # Test that trailing zeros are removed
        self.assertEqual(ftoa(10.0), '10')
        self.assertEqual(ftoa(1), '1')
        self.assertEqual(ftoa(1.0), '1')
        self.assertEqual(ftoa(-1.0), '-1')
        self.assertEqual(ftoa(0.0), '0')
        self.assertEqual(ftoa(1e100), '1e+100')
        self.assertEqual(ftoa(1e-100), '1e-100')

        self.assertEqual(ftoa(10.0, True), '10')
        self.assertEqual(ftoa(1, True), '1')
        self.assertEqual(ftoa(1.0, True), '1')
        self.assertEqual(ftoa(-1.0, True), '(-1)')
        self.assertEqual(ftoa(0.0, True), '0')
        self.assertEqual(ftoa(1e100, True), '1e+100')
        self.assertEqual(ftoa(1e-100, True), '1e-100')

        # Check None
        self.assertIsNone(ftoa(None))

        m = ConcreteModel()
        m.x = Var()
        with self.assertRaisesRegex(
            ValueError, r'Converting non-fixed bound or value to string: 2\*x'
        ):
            self.assertIsNone(ftoa(2 * m.x))

    @unittest.skipIf(not numpy_available, "NumPy is not available")
    def test_ftoa_precision(self):
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.core', logging.WARNING):
            f = np.longdouble('1.1234567890123456789')
            a = ftoa(f)
        self.assertEqual(a, '1.1234567890123457')
        # Depending on the platform, np.longdouble may or may not have
        # higher precision than float:
        if f == float(f):
            test = self.assertNotRegex
        else:
            test = self.assertRegex
        test(
            log.getvalue(),
            '.*Converting 1.1234567890123456789 to string '
            'resulted in loss of precision',
        )

    def test_filedeterminism(self):
        with LoggingIntercept() as LOG:
            a = FileDeterminism(10)
        self.assertEqual(a, FileDeterminism.ORDERED)
        self.assertEqual('', LOG.getvalue())

        self.assertEqual(str(a), 'FileDeterminism.ORDERED')
        self.assertEqual(f"{a}", 'FileDeterminism.ORDERED')

        with LoggingIntercept() as LOG:
            a = FileDeterminism(1)
        self.assertEqual(a, FileDeterminism.SORT_INDICES)
        self.assertIn(
            'FileDeterminism(1) is deprecated.  '
            'Please use FileDeterminism.SORT_INDICES (20)',
            LOG.getvalue().replace('\n', ' '),
        )

        with self.assertRaisesRegex(ValueError, "5 is not a valid FileDeterminism"):
            FileDeterminism(5)

    def test_InvalidNumber(self):
        a = InvalidNumber(-3)
        b = InvalidNumber(5)
        c = InvalidNumber(5)

        self.assertEqual((a + b).value, 2)
        self.assertEqual((a - b).value, -8)
        self.assertEqual((a * b).value, -15)
        self.assertEqual((a / b).value, -0.6)
        self.assertEqual((a**b).value, -(3**5))
        self.assertEqual(abs(a).value, 3)
        self.assertEqual(abs(b).value, 5)
        self.assertEqual((-a).value, 3)
        self.assertEqual((-b).value, -5)

        self.assertEqual((a + 5).value, 2)
        self.assertEqual((a - 5).value, -8)
        self.assertEqual((a * 5).value, -15)
        self.assertEqual((a / 5).value, -0.6)
        self.assertEqual((a**5).value, -(3**5))

        self.assertEqual((-3 + b).value, 2)
        self.assertEqual((-3 - b).value, -8)
        self.assertEqual((-3 * b).value, -15)
        self.assertEqual((-3 / b).value, -0.6)
        self.assertEqual(((-3) ** b).value, -(3**5))

        self.assertTrue(a < b)
        self.assertTrue(a <= b)
        self.assertFalse(a > b)
        self.assertFalse(a >= b)
        self.assertFalse(a == b)
        self.assertTrue(a != b)

        self.assertFalse(c < b)
        self.assertTrue(c <= b)
        self.assertFalse(c > b)
        self.assertTrue(c >= b)
        self.assertTrue(c == b)
        self.assertFalse(c != b)

        self.assertTrue(a < 5)
        self.assertTrue(a <= 5)
        self.assertFalse(a > 5)
        self.assertFalse(a >= 5)
        self.assertFalse(a == 5)
        self.assertTrue(a != 5)

        self.assertTrue(3 < b)
        self.assertTrue(3 <= b)
        self.assertFalse(3 > b)
        self.assertFalse(3 >= b)
        self.assertFalse(3 == b)
        self.assertTrue(3 != b)

        # TODO: eventually these should raise exceptions
        d = InvalidNumber('abc')
        with self.assertRaisesRegex(
            InvalidValueError,
            r"Cannot emit InvalidNumber\(5\) in compiled representation",
        ):
            repr(b)
        with self.assertRaisesRegex(
            InvalidValueError,
            r"Cannot emit InvalidNumber\('abc'\) in compiled representation",
        ):
            repr(d)
        with self.assertRaisesRegex(
            InvalidValueError,
            r"Cannot emit InvalidNumber\(5\) in compiled representation",
        ):
            f'{b}'
        with self.assertRaisesRegex(
            InvalidValueError,
            r"Cannot emit InvalidNumber\('abc'\) in compiled representation",
        ):
            f'{d}'

    def test_apply_operation(self):
        m = ConcreteModel()
        m.x = Var()
        div = 1 / m.x
        mul = m.x * m.x
        exp = m.x ** (1 / 2)

        with LoggingIntercept() as LOG:
            self.assertEqual(apply_node_operation(exp, [4, 1 / 2]), 2)
        self.assertEqual(LOG.getvalue(), "")

        with LoggingIntercept() as LOG:
            ans = apply_node_operation(mul, [float('inf'), 0])
            self.assertIs(type(ans), InvalidNumber)
            self.assertTrue(math.isnan(ans.value))
        self.assertEqual(LOG.getvalue(), "")

        _halt = pyomo.repn.util.HALT_ON_EVALUATION_ERROR
        try:
            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = True
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(ZeroDivisionError, 'division by zero'):
                    apply_node_operation(div, [1, 0])
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/x\n",
            )

            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = False
            with LoggingIntercept() as LOG:
                val = apply_node_operation(div, [1, 0])
                self.assertEqual(str(val), "InvalidNumber(nan)")
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/x\n",
            )

        finally:
            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = _halt

    def test_complex_number_error(self):
        class Visitor(object):
            pass

        visitor = Visitor()

        m = ConcreteModel()
        m.x = Var()
        exp = m.x ** (1 / 2)

        _halt = pyomo.repn.util.HALT_ON_EVALUATION_ERROR
        try:
            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = True
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    InvalidValueError, 'Pyomo Visitor does not support complex numbers'
                ):
                    complex_number_error(1j, visitor, exp)
            self.assertEqual(
                LOG.getvalue(),
                "Complex number returned from expression\n"
                "\tmessage: Pyomo Visitor does not support complex numbers\n"
                "\texpression: x**0.5\n",
            )

            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    InvalidValueError, 'Pyomo Visitor does not support complex numbers'
                ):
                    complex_number_error(1j, visitor, exp, "'(-1)**(0.5)'")
            self.assertEqual(
                LOG.getvalue(),
                "Complex number returned from expression '(-1)**(0.5)'\n"
                "\tmessage: Pyomo Visitor does not support complex numbers\n"
                "\texpression: x**0.5\n",
            )

            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = False
            with LoggingIntercept() as LOG:
                val = complex_number_error(1j, visitor, exp)
                self.assertEqual(str(val), "InvalidNumber(1j)")
            self.assertEqual(
                LOG.getvalue(),
                "Complex number returned from expression\n"
                "\tmessage: Pyomo Visitor does not support complex numbers\n"
                "\texpression: x**0.5\n",
            )

        finally:
            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = _halt

    def test_categorize_valid_components(self):
        m = ConcreteModel()
        m.x = Var()
        m.o = Objective()
        m.b2 = Block()
        m.b2.e = Expression()
        m.b2.p = Param()
        m.b2.q = Param()
        m.b = Block()
        m.b.p = Param()
        m.s = Suffix()
        m.b.t = Suffix()
        m.b.s = Suffix()

        m.b.deactivate()

        component_map, unrecognized = categorize_valid_components(
            m, valid={Var, Block}, targets={Param, Objective, Set}
        )
        self.assertStructuredAlmostEqual(
            component_map, {Param: [m.b2], Objective: [m], Set: []}
        )
        self.assertStructuredAlmostEqual(unrecognized, {Suffix: [m.s]})

        component_map, unrecognized = categorize_valid_components(
            m, active=None, valid={Var, Block}, targets={Param, Objective, Set}
        )
        self.assertStructuredAlmostEqual(
            component_map, {Param: [m.b2, m.b], Objective: [m], Set: []}
        )
        self.assertStructuredAlmostEqual(
            unrecognized, {Suffix: [m.s, m.b.t, m.b.s], Expression: [m.b2.e]}
        )

        component_map, unrecognized = categorize_valid_components(
            m, sort=True, valid={Var, Block}, targets={Param, Objective, Set}
        )
        self.assertStructuredAlmostEqual(
            component_map, {Param: [m.b2], Objective: [m], Set: []}
        )
        self.assertStructuredAlmostEqual(unrecognized, {Suffix: [m.s]})

        component_map, unrecognized = categorize_valid_components(
            m,
            sort=True,
            active=None,
            valid={Var, Block},
            targets={Param, Objective, Set},
        )
        self.assertStructuredAlmostEqual(
            component_map, {Param: [m.b, m.b2], Objective: [m], Set: []}
        )
        self.assertStructuredAlmostEqual(
            unrecognized, {Suffix: [m.s, m.b.s, m.b.t], Expression: [m.b2.e]}
        )

        with self.assertRaises(AssertionError):
            component_map, unrecognized = categorize_valid_components(m, active=False)

        with self.assertRaisesRegex(
            DeveloperError,
            "categorize_valid_components: Cannot have component type "
            r"\[\<class[^>]*Set'\>\] in both the `valid` "
            "and `targets` sets",
        ):
            categorize_valid_components(
                m, valid={Var, Block, Set}, targets={Param, Objective, Set}
            )

    def test_FileDeterminism_to_SortComponents(self):
        self.assertEqual(
            FileDeterminism_to_SortComponents(FileDeterminism(0)),
            SortComponents.unsorted,
        )
        self.assertEqual(
            FileDeterminism_to_SortComponents(FileDeterminism.ORDERED),
            SortComponents.deterministic,
        )
        self.assertEqual(
            FileDeterminism_to_SortComponents(FileDeterminism.SORT_INDICES),
            SortComponents.indices,
        )
        self.assertEqual(
            FileDeterminism_to_SortComponents(FileDeterminism.SORT_SYMBOLS),
            SortComponents.indices | SortComponents.alphabetical,
        )

    def test_initialize_var_map_from_column_order(self):
        class MockConfig(object):
            column_order = None
            file_determinism = FileDeterminism(0)

        m = ConcreteModel()
        m.x = Var()
        m.y = Var([3, 2])
        m.c = Block()
        m.c.x = Var()
        m.c.y = Var([5, 4])
        m.b = Block()
        m.b.x = Var()
        m.b.y = Var([7, 6])

        # No column order, no determinism:
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), []
        )
        # ...sort indices (but not names):
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]],
        )
        # ...sort indices and names:
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]],
        )

        # column order "False", no determinism:
        MockConfig.column_order = False
        MockConfig.file_determinism = FileDeterminism(0)
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()), []
        )
        # ...sort indices (but not names):
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]],
        )
        # ...sort indices and names:
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]],
        )

        # column order "True", no determinism:
        MockConfig.column_order = True
        MockConfig.file_determinism = FileDeterminism(0)
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]],
        )
        # ...sort indices (but not names):
        MockConfig.column_order = True
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]],
        )
        # ...sort indices and names:
        MockConfig.column_order = True
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]],
        )

        # column order "True", no determinism, pre-specified entries
        # (prespecified stay at the beginning of the list):
        MockConfig.column_order = True
        MockConfig.file_determinism = FileDeterminism.ORDERED
        var_map = {id(m.b.y[7]): m.b.y[7], id(m.c.y[5]): m.c.y[5], id(m.y[3]): m.y[3]}
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, var_map).values()),
            [m.b.y[7], m.c.y[5], m.y[3], m.x, m.y[2], m.c.x, m.c.y[4], m.b.x, m.b.y[6]],
        )

        # column order from a ComponentMap
        MockConfig.column_order = ComponentMap(
            (v, i) for i, v in enumerate([m.b.y, m.y, m.c.y[4], m.x])
        )
        MockConfig.file_determinism = FileDeterminism.ORDERED
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.y[5]],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        )

        # column order from a list
        MockConfig.column_order = [m.b.y, m.y, m.c.y[4], m.x]
        ref = list(MockConfig.column_order)
        MockConfig.file_determinism = FileDeterminism.ORDERED
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.y[5]],
        )
        # verify no side effects
        self.assertEqual(MockConfig.column_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        # verify no side effects
        self.assertEqual(MockConfig.column_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        )
        # verify no side effects
        self.assertEqual(MockConfig.column_order, ref)

    def test_ordered_active_constraints(self):
        class MockConfig(object):
            row_order = None
            file_determinism = FileDeterminism(0)

        m = ConcreteModel()
        m.v = Var()
        m.x = Constraint(expr=m.v >= 0)
        m.y = Constraint([3, 2], rule=lambda b, i: m.v >= 0)
        m.c = Block()
        m.c.x = Constraint(expr=m.v >= 0)
        m.c.y = Constraint([5, 4], rule=lambda b, i: m.v >= 0)
        m.b = Block()
        m.b.x = Constraint(expr=m.v >= 0)
        m.b.y = Constraint([7, 6], rule=lambda b, i: m.v >= 0)

        # No row order, no determinism:
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]],
        )
        # ...sort indices (but not names):
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]],
        )
        # ...sort indices and names:
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]],
        )

        # Empty row order, no determinism:
        MockConfig.row_order = []
        MockConfig.file_determinism = FileDeterminism(0)
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]],
        )

        # row order "False", no determinism:
        MockConfig.row_order = False
        MockConfig.file_determinism = FileDeterminism(0)
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]],
        )
        # ...sort indices (but not names):
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]],
        )
        # ...sort indices and names:
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]],
        )

        # row order "True", no determinism:
        MockConfig.row_order = True
        MockConfig.file_determinism = FileDeterminism(0)
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[3], m.y[2], m.c.x, m.c.y[5], m.c.y[4], m.b.x, m.b.y[7], m.b.y[6]],
        )
        # ...sort indices (but not names):
        MockConfig.row_order = True
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[2], m.y[3], m.c.x, m.c.y[4], m.c.y[5], m.b.x, m.b.y[6], m.b.y[7]],
        )
        # ...sort indices and names:
        MockConfig.row_order = True
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.x, m.y[2], m.y[3], m.b.x, m.b.y[6], m.b.y[7], m.c.x, m.c.y[4], m.c.y[5]],
        )

        # row order from a ComponentMap
        MockConfig.row_order = ComponentMap(
            (v, i) for i, v in enumerate([m.b.y, m.y, m.c.y[4], m.x])
        )
        MockConfig.file_determinism = FileDeterminism.ORDERED
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        )

        # row order from a list
        MockConfig.row_order = [m.b.y, m.y, m.c.y[4], m.x]
        ref = list(MockConfig.row_order)
        MockConfig.file_determinism = FileDeterminism.ORDERED
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        # verify no side effects
        self.assertEqual(MockConfig.row_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        # verify no side effects
        self.assertEqual(MockConfig.row_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        )
        # verify no side effects
        self.assertEqual(MockConfig.row_order, ref)

    def test_ExitNodeDispatcher_registration(self):
        end = ExitNodeDispatcher(
            {
                ProductExpression: lambda v, n, d1, d2: d1 * d2,
                Expression: lambda v, n, d: d,
            }
        )
        self.assertEqual(len(end), 2)

        node = ProductExpression((3, 4))
        self.assertEqual(end[node.__class__](None, node, *node.args), 12)
        self.assertEqual(len(end), 2)

        node = Expression(initialize=5)
        node.construct()
        self.assertEqual(end[node.__class__](None, node, *node.args), 5)
        self.assertEqual(len(end), 3)
        self.assertIn(node.__class__, end)

        node = NPV_ProductExpression((6, 7))
        self.assertEqual(end[node.__class__](None, node, *node.args), 42)
        self.assertEqual(len(end), 4)
        self.assertIn(NPV_ProductExpression, end)

        class NewProductExpression(ProductExpression):
            pass

        node = NewProductExpression((6, 7))
        with self.assertRaisesRegex(
            DeveloperError, r".*Unexpected expression node type 'NewProductExpression'"
        ):
            end[node.__class__](None, node, *node.args)
        self.assertEqual(len(end), 4)

        end[SumExpression, 2] = lambda v, n, *d: 2 * sum(d)
        self.assertEqual(len(end), 5)

        node = SumExpression((1, 2, 3))
        self.assertEqual(end[node.__class__, 2](None, node, *node.args), 12)
        self.assertEqual(len(end), 5)

        with self.assertRaisesRegex(
            DeveloperError,
            r"(?s)Base expression key '\(<class.*"
            r"'pyomo.core.expr.numeric_expr.SumExpression'>, 3\)' not found when.*"
            r"inserting dispatcher for node 'SumExpression' while walking.*"
            r"expression tree.",
        ):
            end[node.__class__, 3](None, node, *node.args)
        self.assertEqual(len(end), 5)

        end[SumExpression] = lambda v, n, *d: sum(d)
        self.assertEqual(len(end), 6)
        self.assertIn(SumExpression, end)

        self.assertEqual(end[node.__class__, 1](None, node, *node.args), 6)
        self.assertEqual(len(end), 7)
        self.assertIn((SumExpression, 1), end)

        self.assertEqual(end[node.__class__, 3, 4, 5, 6](None, node, *node.args), 6)
        self.assertEqual(len(end), 7)
        self.assertNotIn((SumExpression, 3, 4, 5, 6), end)

    def test_BeforeChildDispatcher_registration(self):
        class BeforeChildDispatcherTester(BeforeChildDispatcher):
            @staticmethod
            def _before_var(visitor, child):
                return child

            @staticmethod
            def _before_named_expression(visitor, child):
                return child

        class VisitorTester(object):
            def check_constant(self, value, node):
                return value

            def evaluate(self, node):
                return node()

        visitor = VisitorTester()

        bcd = BeforeChildDispatcherTester()
        self.assertEqual(len(bcd), 0)

        node = 5
        self.assertEqual(bcd[node.__class__](None, node), (False, (_CONSTANT, 5)))
        self.assertIs(bcd[int], bcd._before_native)
        self.assertEqual(len(bcd), 1)

        node = 'string'
        ans = bcd[node.__class__](None, node)
        self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber(node))))
        self.assertEqual(
            ''.join(ans[1][1].causes),
            "'string' (<class 'str'>) is not a valid numeric type",
        )
        self.assertIs(bcd[str], bcd._before_string)
        self.assertEqual(len(bcd), 2)

        node = True
        ans = bcd[node.__class__](None, node)
        self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber(node))))
        self.assertEqual(
            ''.join(ans[1][1].causes),
            "True (<class 'bool'>) is not a valid numeric type",
        )
        self.assertIs(bcd[bool], bcd._before_invalid)
        self.assertEqual(len(bcd), 3)

        node = 1j
        ans = bcd[node.__class__](None, node)
        self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber(node))))
        self.assertEqual(
            ''.join(ans[1][1].causes), "Complex number returned from expression"
        )
        self.assertIs(bcd[complex], bcd._before_complex)
        self.assertEqual(len(bcd), 4)

        class new_int(int):
            pass

        node = new_int(5)
        self.assertEqual(bcd[node.__class__](None, node), (False, (_CONSTANT, 5)))
        self.assertIs(bcd[new_int], bcd._before_native)
        self.assertEqual(len(bcd), 5)

        node = []
        ans = bcd[node.__class__](None, node)
        self.assertEqual(ans, (False, (_CONSTANT, InvalidNumber([]))))
        self.assertEqual(
            ''.join(ans[1][1].causes), "[] (<class 'list'>) is not a valid numeric type"
        )
        self.assertIs(bcd[list], bcd._before_invalid)
        self.assertEqual(len(bcd), 6)

        node = Var(initialize=7)
        node.construct()
        self.assertIs(bcd[node.__class__](None, node), node)
        self.assertIs(bcd[node.__class__], bcd._before_var)
        self.assertEqual(len(bcd), 7)

        node = Param(initialize=8)
        node.construct()
        self.assertEqual(bcd[node.__class__](visitor, node), (False, (_CONSTANT, 8)))
        self.assertIs(bcd[node.__class__], bcd._before_param)
        self.assertEqual(len(bcd), 8)

        node = Expression(initialize=9)
        node.construct()
        self.assertIs(bcd[node.__class__](None, node), node)
        self.assertIs(bcd[node.__class__], bcd._before_named_expression)
        self.assertEqual(len(bcd), 9)

        node = SumExpression((3, 5))
        self.assertEqual(bcd[node.__class__](None, node), (True, None))
        self.assertIs(bcd[node.__class__], bcd._before_general_expression)
        self.assertEqual(len(bcd), 10)

        node = NPV_ProductExpression((3, 5))
        self.assertEqual(bcd[node.__class__](visitor, node), (False, (_CONSTANT, 15)))
        self.assertEqual(len(bcd), 12)
        self.assertIs(bcd[NPV_ProductExpression], bcd._before_npv)
        self.assertIs(bcd[ProductExpression], bcd._before_general_expression)
        self.assertEqual(len(bcd), 12)

        node = NPV_DivisionExpression((3, 0))
        self.assertEqual(bcd[node.__class__](visitor, node), (True, None))
        self.assertEqual(len(bcd), 14)
        self.assertIs(bcd[NPV_DivisionExpression], bcd._before_npv)
        self.assertIs(bcd[DivisionExpression], bcd._before_general_expression)
        self.assertEqual(len(bcd), 14)


if __name__ == "__main__":
    unittest.main()
