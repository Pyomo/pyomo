# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
import re
import sys
import random
from io import StringIO

import pyomo.common.unittest as unittest
import unittest.mock as mock

from pyomo.common.dependencies import dill_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.fileutils import this_file_dir

from pyomo.environ import (
    TransformationFactory,
    Block,
    Set,
    Constraint,
    Var,
    RealSet,
    ComponentMap,
    value,
    log,
    ConcreteModel,
    Any,
    Suffix,
    SolverFactory,
    RangeSet,
    Param,
    Objective,
    TerminationCondition,
    NonNegativeReals,
)
from pyomo.core.expr.compare import (
    assertExpressionsEqual,
    assertExpressionsStructurallyEqual,
)
import pyomo.core.expr as EXPR
from pyomo.core.base import constraint
from pyomo.repn import generate_standard_repn
from pyomo.repn.linear import LinearRepnVisitor
from pyomo.repn.quadratic import QuadraticRepnVisitor
from pyomo.repn.util import OrderedVarRecorder
from pyomo.core.base import SortComponents
from pyomo.common.dependencies import networkx_available
import pyomo.common.unittest as unittest

from pyomo.gdp import Disjunct, Disjunction, GDP_Error
import pyomo.gdp.plugins.hull as hull_module
import pyomo.gdp.tests.models as models
import pyomo.gdp.tests.common_tests as ct

from pyomo.gdp.plugins.reverse_polar_enumeration_cuts import get_constraint

# The pyomo expression system collapses 1.0*x into x, which results in
# floating-point 1.0 being converted to integer 1 when
# assertExpressionsEqual looks at them, and those values are not the
# same regardless of the value of `places`. Work around this.
ALMOST_ONE = 1.00000000000001


class TestReversePolarEnumerationCuts(unittest.TestCase):
    @unittest.skipUnless(networkx_available, "Networkx is not available")
    def test_example(self):
        # Example model and first cut from Connor's poster
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 20))
        m.x2 = Var(bounds=(0, 20))
        m.x3 = Var(bounds=(0, 20))
        m.x4 = Var(bounds=(0, 20))

        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=5 * m.x1 - 3 * m.x2 + m.x4 >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=3 * m.x1 - m.x2 + 2 * m.x3 - 3 * m.x4 >= 1)
        m.d3 = Disjunct()
        m.d3.c = Constraint(expr=4 * m.x1 - 6 * m.x2 + 4 * m.x3 - 2 * m.x4 >= 1)
        m.d4 = Disjunct()
        m.d4.c = Constraint(expr=2 * m.x1 - 2 * m.x2 - 2 * m.x3 >= 1)
        m.d = Disjunction(expr=[m.d1, m.d2, m.d3, m.d4])

        TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)
        cons = get_constraint(m, m.d)
        assertExpressionsEqual(
            self,
            cons[0].body,
            5.0 * m.x1 + m.x4 + 4.0 * m.x3 - (5.0 / 3.0) * m.x2,
            places=8,
        )

    @unittest.skipUnless(networkx_available, "Networkx is not available")
    def test_linearly_many_easy(self):
        # Easier version of below: 4 variables, 3 cuts
        # After preprocessing the disjunction should look exactly the same,
        # and the obtained cuts should be:
        # x1 + x2 - x3 - x4 >= 1
        # 3x1 + x2 -3x3 - 3x4 >= 1
        # 4x1 + x2 -3x3 - 4x4 >= 1
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 20))
        m.x2 = Var(bounds=(0, 20))
        m.x3 = Var(bounds=(0, 20))
        m.x4 = Var(bounds=(0, 20))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x1 - m.x3 - m.x4 >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x2 - 3 * m.x3 - 4 * m.x4 >= 1)
        m.d = Disjunction(expr=[m.d1, m.d2])

        TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)
        cons = get_constraint(m, m.d)

        self.assertEqual(3, len(cons))

        assertExpressionsEqual(
            self,
            cons[0].body,
            m.x1 + m.x2 - ALMOST_ONE * m.x3 - ALMOST_ONE * m.x4,
            places=8,
        )
        self.assertEqual(1, cons[0].lower)
        self.assertIsNone(cons[0].upper)
        assertExpressionsEqual(
            self, cons[1].body, 3.0 * m.x1 + m.x2 - 3.0 * m.x3 - 3.0 * m.x4, places=8
        )
        self.assertEqual(1, cons[1].lower)
        self.assertIsNone(cons[1].upper)
        assertExpressionsEqual(
            self, cons[2].body, 4.0 * m.x1 + m.x2 - 3.0 * m.x3 - 4.0 * m.x4, places=8
        )
        self.assertEqual(1, cons[2].lower)
        self.assertIsNone(cons[2].upper)

    @unittest.skipUnless(networkx_available, "Networkx is not available")
    def test_linearly_many_medium(self):
        # Constructing the model according to this pattern with n
        # variables, the resulting model should have exactly n/2 + 1
        # cuts. The intermediate disjunction formed during preprocessing
        # has 4 disjuncts
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 20))
        m.x2 = Var(bounds=(0, 20))
        m.x3 = Var(bounds=(0, 20))
        m.x4 = Var(bounds=(0, 20))
        m.x5 = Var(bounds=(0, 20))
        m.x6 = Var(bounds=(0, 20))
        m.x7 = Var(bounds=(0, 20))
        m.x8 = Var(bounds=(0, 20))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x1 + m.x2 - m.x5 - m.x6 - m.x7 - m.x8 >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(
            expr=m.x3 + m.x4 - 5 * m.x5 - 6 * m.x6 - 7 * m.x7 - 8 * m.x8 >= 1
        )
        m.d = Disjunction(expr=[m.d1, m.d2])

        TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)
        cons = get_constraint(m, m.d)

        self.assertEqual(5, len(cons))
        assertExpressionsEqual(
            self,
            cons[0].body,
            m.x1
            + m.x2
            + m.x3
            + m.x4
            - ALMOST_ONE * m.x5
            - ALMOST_ONE * m.x6
            - ALMOST_ONE * m.x7
            - ALMOST_ONE * m.x8,
            places=8,
        )
        self.assertEqual(1, cons[0].lower)
        self.assertIsNone(cons[0].upper)
        assertExpressionsEqual(
            self,
            cons[1].body,
            5.0 * m.x1
            + 5.0 * m.x2
            + m.x3
            + m.x4
            - 5.0 * m.x5
            - 5.0 * m.x6
            - 5.0 * m.x7
            - 5.0 * m.x8,
            places=8,
        )
        self.assertEqual(1, cons[1].lower)
        self.assertIsNone(cons[1].upper)
        assertExpressionsEqual(
            self,
            cons[2].body,
            6.0 * m.x1
            + 6.0 * m.x2
            + m.x3
            + m.x4
            - 5.0 * m.x5
            - 6.0 * m.x6
            - 6.0 * m.x7
            - 6.0 * m.x8,
            places=8,
        )
        self.assertEqual(1, cons[2].lower)
        self.assertIsNone(cons[2].upper)
        assertExpressionsEqual(
            self,
            cons[3].body,
            7.0 * m.x1
            + 7.0 * m.x2
            + m.x3
            + m.x4
            - 5.0 * m.x5
            - 6.0 * m.x6
            - 7.0 * m.x7
            - 7.0 * m.x8,
            places=8,
        )
        self.assertEqual(1, cons[3].lower)
        self.assertIsNone(cons[3].upper)
        assertExpressionsEqual(
            self,
            cons[4].body,
            8.0 * m.x1
            + 8.0 * m.x2
            + m.x3
            + m.x4
            - 5.0 * m.x5
            - 6.0 * m.x6
            - 7.0 * m.x7
            - 8.0 * m.x8,
            places=8,
        )
        self.assertEqual(1, cons[4].lower)
        self.assertIsNone(cons[4].upper)

    @unittest.skipUnless(networkx_available, "Networkx is not available")
    def test_linearly_many_hard(self):
        # As before but with N a large(ish) number divisible by 4
        m = ConcreteModel()
        N = 40
        m.idx = RangeSet(1, N)
        m.x = Var(m.idx, bounds=(0, 20))
        m.d1 = Disjunct()
        m.d1.c = Constraint(
            expr=sum(m.x[i] for i in range(1, round(N / 4) + 1))
            - sum(m.x[i] for i in range(round(N / 2) + 1, N + 1))
            >= 1
        )
        m.d2 = Disjunct()
        m.d2.c = Constraint(
            expr=sum(m.x[i] for i in range(round(N / 4) + 1, round(N / 2) + 1))
            - sum(i * m.x[i] for i in range(round(N / 2) + 1, N + 1))
            >= 1
        )
        m.d = Disjunction(expr=[m.d1, m.d2])

        TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)
        cons = get_constraint(m, m.d)

        self.assertEqual(N / 2 + 1, len(cons))

    @unittest.skipUnless(networkx_available, "Networkx is not available")
    def test_not_Jm_or_Jp(self):
        # Make sure we don't choke when a variable has negative and zero
        # coefficients, but no positive ones.
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 20))
        m.x2 = Var(bounds=(0, 20))
        m.x3 = Var(bounds=(0, 20))
        m.x4 = Var(bounds=(0, 20))
        m.x5 = Var(bounds=(0, 20))
        m.x6 = Var(bounds=(0, 20))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x1 - m.x4 - 2 * m.x5 - 2 * m.x6 >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x2 - 3 * m.x5 - 2 * m.x6 >= 1)
        m.dn = Disjunction(expr=[m.d1, m.d2])
        # does not raise exception
        TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)

    @unittest.skipUnless(networkx_available, "Networkx is not available")
    def test_exponentially_many(self):
        # For this pattern, a disjunction on n variables with n/2
        # disjuncts leads to a total of 2^{n/2} - 1 cuts. Here we have
        # n=6 so there should be 7 cuts generated.
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 20))
        m.x2 = Var(bounds=(0, 20))
        m.x3 = Var(bounds=(0, 20))
        m.x4 = Var(bounds=(0, 20))
        m.x5 = Var(bounds=(0, 20))
        m.x6 = Var(bounds=(0, 20))
        m.d1 = Disjunct()
        m.d1.c = Constraint(expr=m.x1 - m.x4 - 2 * m.x5 - 2 * m.x6 >= 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x2 - 2 * m.x4 - m.x5 - 2 * m.x6 >= 1)
        m.d3 = Disjunct()
        m.d3.c = Constraint(expr=m.x3 - 2 * m.x4 - 2 * m.x5 - m.x6 >= 1)
        m.dn = Disjunction(expr=[m.d1, m.d2, m.d3])

        TransformationFactory('gdp.reverse_polar_enumeration_cuts').apply_to(m)
        cons = get_constraint(m, m.dn)

        self.assertEqual(7, len(cons))
        assertExpressionsEqual(
            self,
            cons[0].body,
            m.x1
            + m.x2
            + m.x3
            - ALMOST_ONE * m.x4
            - ALMOST_ONE * m.x5
            - ALMOST_ONE * m.x6,
            places=8,
        )
        assertExpressionsEqual(
            self,
            cons[1].body,
            2.0 * m.x1
            + m.x2
            + m.x3
            - 2.0 * m.x4
            - ALMOST_ONE * m.x5
            - ALMOST_ONE * m.x6,
            places=8,
        )
        assertExpressionsEqual(
            self,
            cons[2].body,
            m.x1
            + 2.0 * m.x2
            + m.x3
            - ALMOST_ONE * m.x4
            - 2.0 * m.x5
            - ALMOST_ONE * m.x6,
            places=8,
        )
        assertExpressionsEqual(
            self,
            cons[3].body,
            m.x1
            + m.x2
            + 2.0 * m.x3
            - ALMOST_ONE * m.x4
            - ALMOST_ONE * m.x5
            - 2.0 * m.x6,
            places=8,
        )
        assertExpressionsEqual(
            self,
            cons[4].body,
            2.0 * m.x1
            + 2.0 * m.x2
            + m.x3
            - 2.0 * m.x4
            - 2.0 * m.x5
            - ALMOST_ONE * m.x6,
            places=8,
        )
        assertExpressionsEqual(
            self,
            cons[5].body,
            2.0 * m.x1
            + m.x2
            + 2.0 * m.x3
            - 2.0 * m.x4
            - ALMOST_ONE * m.x5
            - 2.0 * m.x6,
            places=8,
        )
        assertExpressionsEqual(
            self,
            cons[6].body,
            m.x1
            + 2.0 * m.x2
            + 2.0 * m.x3
            - ALMOST_ONE * m.x4
            - 2.0 * m.x5
            - 2.0 * m.x6,
            places=8,
        )
