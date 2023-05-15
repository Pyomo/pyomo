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
from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
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
    ftoa,
    FileDeterminism,
    categorize_valid_components,
    apply_node_operation,
    FileDeterminism_to_SortComponents,
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
                ValueError, r'Converting non-fixed bound or value to string: 2\*x'):
            self.assertIsNone(ftoa(2*m.x))

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
            test = self.assertNotRegexpMatches
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

    def test_apply_operation(self):
        m = ConcreteModel()
        m.x = Var()
        div = 1 / m.x
        exp = m.x ** (1 / 2)

        with LoggingIntercept() as LOG:
            self.assertEqual(apply_node_operation(exp, [4, 1 / 2]), 2)
        self.assertEqual(LOG.getvalue(), "")

        _halt = pyomo.repn.util.HALT_ON_EVALUATION_ERROR
        try:
            pyomo.repn.util.HALT_ON_EVALUATION_ERROR = True
            with LoggingIntercept() as LOG:
                with self.assertRaisesRegex(
                    ValueError, 'Pyomo does not support complex numbers'
                ):
                    print(apply_node_operation(exp, [-1, 1 / 2]))
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'pow(-1, 0.5)'\n"
                "\tmessage: Pyomo does not support complex numbers\n"
                "\texpression: x**0.5\n",
            )

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
                val = apply_node_operation(exp, [-1, 1 / 2])
                self.assertTrue(math.isnan(val), f"{val} is not NaN")
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'pow(-1, 0.5)'\n"
                "\tmessage: Pyomo does not support complex numbers\n"
                "\texpression: x**0.5\n",
            )

            with LoggingIntercept() as LOG:
                val = apply_node_operation(div, [1, 0])
                self.assertTrue(math.isnan(val), f"{val} is not NaN")
            self.assertEqual(
                LOG.getvalue(),
                "Exception encountered evaluating expression 'div(1, 0)'\n"
                "\tmessage: division by zero\n"
                "\texpression: 1/x\n",
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
            SortComponents.unsorted,
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
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        )

        # column order from a list
        MockConfig.column_order = [m.b.y, m.y, m.c.y[4], m.x]
        ref = list(MockConfig.column_order)
        MockConfig.file_determinism = FileDeterminism.ORDERED
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x],
        )
        # verify no side effects
        self.assertEqual(MockConfig.column_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_INDICES
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        # verify no side effects
        self.assertEqual(MockConfig.column_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        self.assertEqual(
            list(initialize_var_map_from_column_order(m, MockConfig, {}).values()),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
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
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
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
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.c.x, m.c.y[5], m.b.x],
        )
        # verify no side effects
        self.assertEqual(MockConfig.row_order, ref)
        MockConfig.file_determinism = FileDeterminism.SORT_SYMBOLS
        # TODO: this should be the correct baseline, but resolving it requires PR#2829
        # [m.b.y[6], m.b.y[7], m.y[2], m.y[3], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        self.assertEqual(
            list(ordered_active_constraints(m, MockConfig)),
            [m.b.y[7], m.b.y[6], m.y[3], m.y[2], m.c.y[4], m.x, m.b.x, m.c.x, m.c.y[5]],
        )
        # verify no side effects
        self.assertEqual(MockConfig.row_order, ref)


if __name__ == "__main__":
    unittest.main()
