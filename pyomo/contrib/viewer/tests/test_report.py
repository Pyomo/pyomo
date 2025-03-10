#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  ___________________________________________________________________________
#
#  This module was originally developed as part of the IDAES PSE Framework
#
#  Institute for the Design of Advanced Energy Systems Process Systems
#  Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
#  software owners: The Regents of the University of California, through
#  Lawrence Berkeley National Laboratory,  National Technology & Engineering
#  Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
#  University Research Corporation, et al. All rights reserved.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
Test data model items for QTreeView. These tests should work even without PyQt.
"""

import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.environ import (
    ConcreteModel,
    Var,
    BooleanVar,
    Block,
    Param,
    Expression,
    Constraint,
    Objective,
    ExternalFunction,
    Reals,
    log,
    sin,
    sqrt,
    expr,
)
import pyomo.contrib.viewer.report as rpt
import pyomo.contrib.viewer.ui_data as uidata


class TestReportFunctions(unittest.TestCase):
    def setUp(self):
        # Borrowed this test model from the trust region tests
        m = ConcreteModel()
        m.z = Var(range(3), domain=Reals, initialize=2.0)
        m.x = Var(range(4), initialize=2.0)
        m.x[1] = 1.0
        m.x[2] = 0.0
        m.x[3] = None

        m.b1 = Block()
        m.b1.e1 = Expression(expr=m.x[0] + m.x[1])
        m.b1.e2 = Expression(expr=m.x[0] / m.x[2])
        m.b1.e3 = Expression(expr=m.x[3] * m.x[1])
        m.b1.e4 = Expression(expr=log(m.x[2]))
        m.b1.e5 = Expression(expr=log(m.x[2] - 2))

        def blackbox(a, b):
            return sin(a - b)

        self.bb = ExternalFunction(blackbox)

        m.obj = Objective(
            expr=(m.z[0] - 1.0) ** 2
            + (m.z[0] - m.z[1]) ** 2
            + (m.z[2] - 1.0) ** 2
            + (m.x[0] - 1.0) ** 4
            + (m.x[1] - 1.0) ** 6  # + m.bb(m.x[0],m.x[1])
        )
        m.c1 = Constraint(
            expr=m.x[0] * m.z[0] ** 2 + self.bb(m.x[0], m.x[1]) == 2 * sqrt(2.0)
        )
        m.c2 = Constraint(expr=m.z[2] ** 4 * m.z[1] ** 2 + m.z[1] == 8 + sqrt(2.0))
        m.c3 = Constraint(expr=m.x[1] == 3)
        m.c4 = Constraint(expr=0 == 3 / m.x[2])
        m.c5 = Constraint(expr=0 == log(m.x[2]))
        m.c6 = Constraint(expr=0 == log(m.x[2] - 4))
        m.c7 = Constraint(expr=0 == log(m.x[3]))
        m.p1 = Param(mutable=True, initialize=1)
        m.c8 = Constraint(expr=m.x[1] <= 1 / m.p1)
        m.c8b = Constraint(expr=m.x[1] >= 1 / m.p1)
        m.c9 = Constraint(expr=m.x[1] <= 1)
        m.c10 = Constraint(expr=m.x[1] >= 1)
        m.p1 = 0
        self.m = m.clone()

    def test_value_no_exception(self):
        # Try to divide by zero
        self.m.x[2] = 0
        v = rpt.value_no_exception(self.m.b1.e2, div0="I like to divide by zero")
        assert v == "I like to divide by zero"
        # Try as calculation with None
        self.m.x[2] = None
        v = rpt.value_no_exception(self.m.b1.e2, div0=None)
        assert v is None
        # Try log of negative number
        self.m.x[2] = 0.0
        v = rpt.value_no_exception(self.m.b1.e5)
        assert v is None
        # Try a valid calculation
        self.m.x[2] = 2.0
        v = rpt.value_no_exception(self.m.b1.e2, div0=None)
        self.assertAlmostEqual(v, 1)

    def test_get_residual(self):
        dat = uidata.UIData(self.m)
        # so that the model viewer doesn't run slow on large models,
        # you have to explicitly ask for constraints and expressions
        # to be calculated. Getting the residual before calculation
        # should just give None
        assert rpt.get_residual(dat, self.m.c3) is None
        dat.calculate_constraints()
        self.assertAlmostEqual(rpt.get_residual(dat, self.m.c3), 2.0)
        # In c8 the bound has a divide by 0, I think this is only possible
        # with a mutable param
        assert rpt.get_residual(dat, self.m.c8) == "Divide_by_0"
        assert rpt.get_residual(dat, self.m.c8b) == "Divide_by_0"
        self.m.x[2] = 0
        assert rpt.get_residual(dat, self.m.c4) == "Divide_by_0"
        self.m.x[2] = 2
        # haven't recalculated so still error
        assert rpt.get_residual(dat, self.m.c4) == "Divide_by_0"
        dat.calculate_constraints()
        self.assertAlmostEqual(rpt.get_residual(dat, self.m.c4), 3.0 / 2.0)
        self.assertAlmostEqual(rpt.get_residual(dat, self.m.c9), 0)
        self.assertAlmostEqual(rpt.get_residual(dat, self.m.c10), 0)

    def test_active_equalities(self):
        eq = [
            self.m.c1,
            self.m.c2,
            self.m.c3,
            self.m.c4,
            self.m.c5,
            self.m.c6,
            self.m.c7,
        ]
        for i, o in enumerate(rpt.active_equalities(self.m)):
            assert o == eq[i]

    def test_active_constraint_set(self):
        self.m.c4.deactivate()
        assert rpt.active_constraint_set(self.m) == ComponentSet(
            [
                self.m.c1,
                self.m.c2,
                self.m.c3,
                self.m.c5,
                self.m.c6,
                self.m.c7,
                self.m.c8,
                self.m.c8b,
                self.m.c9,
                self.m.c10,
            ]
        )
        self.m.c4.activate()
        assert rpt.active_constraint_set(self.m) == ComponentSet(
            [
                self.m.c1,
                self.m.c2,
                self.m.c3,
                self.m.c4,
                self.m.c5,
                self.m.c6,
                self.m.c7,
                self.m.c8,
                self.m.c8b,
                self.m.c9,
                self.m.c10,
            ]
        )

    def test_active_equality_set(self):
        self.m.c4.deactivate()
        assert rpt.active_equality_set(self.m) == ComponentSet(
            [self.m.c1, self.m.c2, self.m.c3, self.m.c5, self.m.c6, self.m.c7]
        )
        self.m.c4.activate()
        assert rpt.active_equality_set(self.m) == ComponentSet(
            [
                self.m.c1,
                self.m.c2,
                self.m.c3,
                self.m.c4,
                self.m.c5,
                self.m.c6,
                self.m.c7,
            ]
        )

    def test_count_free_variables(self):
        assert rpt.count_free_variables(self.m) == 7

    def test_count_equality_constraints(self):
        assert rpt.count_equality_constraints(self.m) == 7

    def test_count_constraints(self):
        assert rpt.count_constraints(self.m) == 11

    def test_degrees_of_freedom(self):
        assert rpt.degrees_of_freedom(self.m) == 0
