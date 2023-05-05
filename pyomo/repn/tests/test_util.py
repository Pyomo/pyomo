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

import pyomo.common.unittest as unittest
from io import StringIO

from pyomo.common.errors import DeveloperError
from pyomo.common.log import LoggingIntercept
from pyomo.environ import (
    ConcreteModel,
    Block,
    Var,
    Param,
    Objective,
    Suffix,
    Expression,
    Set,
)
from pyomo.repn.util import ftoa, FileDeterminism, categorize_valid_components

try:
    import numpy as np

    numpy_available = True
except:
    numpy_available = False


class TestRepnUtils(unittest.TestCase):
    def test_ftoa(self):
        # Test that trailing zeros are removed
        f = 1.0
        a = ftoa(f)
        self.assertEqual(a, '1')

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

    def test_filedeterminism_missing(self):
        with LoggingIntercept() as LOG:
            a = FileDeterminism(10)
        self.assertEqual(a, FileDeterminism.ORDERED)
        self.assertEqual('', LOG.getvalue())

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


if __name__ == "__main__":
    unittest.main()
