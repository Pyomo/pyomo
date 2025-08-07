#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import io
import logging

import pyomo.common.unittest as unittest

from pyomo.common.log import LoggingIntercept
from pyomo.common.tee import capture_output
from pyomo.core.expr import InequalityExpression
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.environ import (
    AbstractModel,
    BooleanConstant,
    BooleanVar,
    ConcreteModel,
    LogicalConstraint,
    LogicalConstraintList,
    TransformationFactory,
    Constraint,
    Param,
    NonNegativeIntegers,
)
from pyomo.repn import generate_standard_repn
from pyomo.gdp import Disjunction


class TestLogicalConstraint(unittest.TestCase):
    def create_model(self, abstract=False):
        if abstract is True:
            model = AbstractModel()
        else:
            model = ConcreteModel()
        model.x = BooleanVar()
        model.y = BooleanVar()
        model.z = BooleanVar()
        return model

    def test_abstract_and_empty(self):
        m = self.create_model(True)

        @m.LogicalConstraint()
        def p(m):
            return LogicalConstraint.Skip

        with self.assertRaisesRegex(
            RuntimeError,
            "Cannot access property 'expr' on AbstractScalarLogicalConstraint "
            "'p' before it has been constructed",
        ):
            m.p.expr

        i = m.create_instance()
        with self.assertRaisesRegex(
            ValueError,
            "Accessing the expr of ScalarLogicalConstraint "
            "'p' before the LogicalConstraint has been assigned "
            "an expression.",
        ):
            i.p.body

    def test_construct(self):
        model = self.create_model()

        @model.LogicalConstraint()
        def p(model):
            return model.x

        self.assertIs(model.p.body, model.x)

        with LoggingIntercept() as LOG:
            with self.assertRaisesRegex(
                IndexError,
                "LogicalConstraint 'q': Cannot initialize multiple indices of "
                "a constraint with a single expression",
            ):
                model.q = LogicalConstraint(range(2), expr=model.x)
        self.assertEqual(
            LOG.getvalue().strip(),
            """
Rule failed when generating expression for LogicalConstraint q with index None:
IndexError: LogicalConstraint 'q': Cannot initialize multiple indices of a constraint with a single expression
Constructing component 'q' from data=None failed:
    IndexError: LogicalConstraint 'q': Cannot initialize multiple indices of a constraint with a single expression
""".strip(),
        )

        with LoggingIntercept(level=logging.DEBUG, module='pyomo.core.base') as LOG:
            model.r = LogicalConstraint()
        self.assertEqual(LOG.getvalue(), 'Constructing logical constraint r\n')

        with LoggingIntercept(level=logging.DEBUG, module='pyomo.core.base') as LOG:
            model.r.construct()
        self.assertEqual(LOG.getvalue(), "")

    def test_indexed_constructor(self):
        m = self.create_model()
        m.p = LogicalConstraint(
            range(7),
            rule=[
                LogicalConstraint.Feasible,
                LogicalConstraint.Infeasible,
                LogicalConstraint.Skip,
                m.x.implies(m.y),
                m.x,
                True,
                False,
            ],
        )

        self.assertExpressionsStructurallyEqual(m.p[0].expr, LogicalConstraint.Feasible)
        self.assertExpressionsStructurallyEqual(
            m.p[1].expr, LogicalConstraint.Infeasible
        )
        self.assertNotIn(2, m.p)
        with self.assertRaises(KeyError):
            m.p[2].expr
        self.assertExpressionsStructurallyEqual(m.p[3].expr, m.x.implies(m.y))
        self.assertExpressionsStructurallyEqual(m.p[4].expr, m.x)
        self.assertExpressionsStructurallyEqual(m.p[5].expr, BooleanConstant(True))
        self.assertExpressionsStructurallyEqual(m.p[6].expr, BooleanConstant(False))

    def test_display(self):
        m = self.create_model()
        m.p = LogicalConstraint(
            range(7),
            rule=[
                LogicalConstraint.Feasible,
                LogicalConstraint.Infeasible,
                LogicalConstraint.Skip,
                m.x.implies(m.y),
                m.x,
                True,
                False,
            ],
        )

        ref = """p : Size=6
    Key : Body
      0 :  True
      1 : False
      3 :  None
      4 :  None
      5 :  True
      6 : False
"""
        OUT = io.StringIO()
        m.p.display(ostream=OUT)
        self.assertEqual(ref, OUT.getvalue())

        with capture_output() as OUT:
            m.p.display()
        self.assertEqual(ref, OUT.getvalue())

        m.x = True
        m.y = False

        OUT = io.StringIO()
        m.p.display(ostream=OUT)
        self.assertEqual(
            """p : Size=6
    Key : Body
      0 :  True
      1 : False
      3 : False
      4 :  True
      5 :  True
      6 : False
""",
            OUT.getvalue(),
        )

        m.p.deactivate()

        OUT = io.StringIO()
        m.p.display(ostream=OUT)
        self.assertEqual("", OUT.getvalue())

    def test_pprint(self):
        m = self.create_model()
        m.p = LogicalConstraint(
            range(7),
            rule=[
                LogicalConstraint.Feasible,
                LogicalConstraint.Infeasible,
                LogicalConstraint.Skip,
                m.x.implies(m.y),
                m.x,
                True,
                False,
            ],
        )

        OUT = io.StringIO()
        m.p.pprint(ostream=OUT)
        self.assertEqual(
            """p : Size=6, Index={0, 1, 2, 3, 4, 5, 6}, Active=True
    Key : Body       : Active
      0 :   Feasible :   True
      1 : Infeasible :   True
      3 :    x --> y :   True
      4 :          x :   True
      5 :       True :   True
      6 :      False :   True
""",
            OUT.getvalue(),
        )

    def test_indexed_deferred_constructor(self):
        m = self.create_model()

        def p_rule(m, i):
            return [
                LogicalConstraint.Feasible,
                LogicalConstraint.Infeasible,
                LogicalConstraint.Skip,
                m.x.implies(m.y),
                m.x,
                True,
                False,
            ][i]

        m.p = LogicalConstraint(NonNegativeIntegers, rule=p_rule)

        self.assertExpressionsStructurallyEqual(m.p[0].expr, LogicalConstraint.Feasible)
        self.assertExpressionsStructurallyEqual(
            m.p[1].expr, LogicalConstraint.Infeasible
        )
        self.assertNotIn(2, m.p)
        with self.assertRaises(KeyError):
            m.p[2].expr
        self.assertExpressionsStructurallyEqual(m.p[3].expr, m.x.implies(m.y))
        self.assertExpressionsStructurallyEqual(m.p[4].expr, m.x)
        self.assertExpressionsStructurallyEqual(m.p[5].expr, BooleanConstant(True))
        self.assertExpressionsStructurallyEqual(m.p[6].expr, BooleanConstant(False))

        m.q = LogicalConstraint(NonNegativeIntegers)
        with self.assertRaises(KeyError):
            m.q[2].expr

    def test_add(self):
        m = self.create_model()
        m.p = LogicalConstraint()
        expr = m.x.implies(m.y)
        with self.assertRaisesRegex(
            ValueError,
            "ScalarLogicalConstraint object 'p' does not accept "
            "index values other than None. Invalid value: 1",
        ):
            m.p.add(1, expr)
        self.assertEqual(len(m.p), 0)
        m.p.add(None, expr)
        self.assertEqual(len(m.p), 1)
        self.assertIs(m.p.expr, expr)

        m.q = LogicalConstraint(range(2))
        m.q.add(0, expr)
        self.assertEqual(len(m.q), 1)
        self.assertIs(m.q[0].expr, expr)
        m.q.add(1, expr)
        self.assertEqual(len(m.q), 2)
        self.assertIs(m.q[1].expr, expr)

    def test_set_value(self):
        m = self.create_model()

        m.p = LogicalConstraint()
        self.assertEqual(len(m.p), 0)

        expr = m.x.implies(m.y)
        m.p = expr
        self.assertIs(m.p.expr, expr)

        with self.assertRaisesRegex(
            ValueError, "LogicalConstraint 'p': rule returned None"
        ):
            m.p = None
        # (test that the expr is unchanged on error)
        self.assertIs(m.p.expr, expr)

        m.p = LogicalConstraint.Skip
        self.assertEqual(len(m.p), 0)

        m.p = LogicalConstraint.Feasible
        self.assertExpressionsStructurallyEqual(m.p.expr, LogicalConstraint.Feasible)

        m.p = LogicalConstraint.Infeasible
        self.assertExpressionsStructurallyEqual(m.p.expr, LogicalConstraint.Infeasible)

        with self.assertRaisesRegex(
            ValueError, "Assigning improper value to LogicalConstraint 'p'."
        ):
            m.p = LogicalConstraint

        with self.assertRaisesRegex(
            ValueError, "Assigning improper value to LogicalConstraint 'p'."
        ):
            m.p = {}

        with self.assertRaisesRegex(
            ValueError, "Assigning improper value to LogicalConstraint 'p'."
        ):
            m.p = Param(mutable=True) + 1

    def test_get_value(self):
        m = self.create_model()
        m.p = LogicalConstraint(expr=m.x.implies(m.y))
        self.assertIs(m.p.get_value(), m.p.expr)

    def check_lor_on_disjunct(self, model, disjunct, x1, x2):
        x1 = x1.get_associated_binary()
        x2 = x2.get_associated_binary()
        disj0 = disjunct.logic_to_linear
        self.assertEqual(len(disj0.component_map(Constraint)), 1)
        lor = disj0.transformed_constraints[1]
        self.assertEqual(lor.lower, 1)
        self.assertIsNone(lor.upper)
        repn = generate_standard_repn(lor.body)
        self.assertEqual(repn.constant, 0)
        self.assertTrue(repn.is_linear())
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], x1)
        self.assertIs(repn.linear_vars[1], x2)
        self.assertEqual(repn.linear_coefs[0], 1)
        self.assertEqual(repn.linear_coefs[1], 1)

    @unittest.skipUnless(sympy_available, "Sympy not available")
    def test_statement_in_Disjunct_with_logical_to_linear(self):
        # This is an old test that originally tested that GDP's
        # BigM/Hull correctly handled Disjuncts with LogicalConstraints
        # (implicitly calling logical_to_linear to leave the transformed
        # algebraic constraints on the Disjuncts).  That is no longer
        # the default behavior.  However, we will preserve this (with an
        # explicit call to logical_to_linear) for posterity
        model = self.create_model()
        model.disj = Disjunction(expr=[[model.x.lor(model.y)], [model.y.lor(model.z)]])

        TransformationFactory('core.logical_to_linear').apply_to(
            model, targets=model.disj.disjuncts
        )

        bigmed = TransformationFactory('gdp.bigm').create_using(model)
        # check that the algebraic versions are living on the Disjuncts
        self.check_lor_on_disjunct(bigmed, bigmed.disj.disjuncts[0], bigmed.x, bigmed.y)
        self.check_lor_on_disjunct(bigmed, bigmed.disj.disjuncts[1], bigmed.y, bigmed.z)

        TransformationFactory('gdp.hull').apply_to(model)
        self.check_lor_on_disjunct(model, model.disj.disjuncts[0], model.x, model.y)
        self.check_lor_on_disjunct(model, model.disj.disjuncts[1], model.y, model.z)

    # TODO look to test_con.py for inspiration
    def test_deprecated_rule_attribute(self):
        def rule(m):
            return m.x.implies(m.x)

        def new_rule(m):
            return m.x.implies(~m.x)

        m = ConcreteModel()
        m.x = BooleanVar()
        m.con = LogicalConstraint(rule=rule)

        self.assertIs(m.con.rule._fcn, rule)
        with LoggingIntercept() as LOG:
            m.con.rule = new_rule
        self.assertIn(
            "DEPRECATED: The 'LogicalConstraint.rule' attribute will be made "
            "read-\nonly",
            LOG.getvalue(),
        )
        self.assertIs(m.con.rule, new_rule)


class TestLogicalConstraintList(unittest.TestCase):
    def create_model(self, abstract=False):
        if abstract is True:
            model = AbstractModel()
        else:
            model = ConcreteModel()
        model.x = BooleanVar()
        model.y = BooleanVar()
        model.z = BooleanVar()
        return model

    def test_construct(self):
        m = self.create_model()

        @m.LogicalConstraintList(starting_index=0)
        def p(m):
            yield m.x
            yield m.x.implies(m.y)
            yield True
            yield LogicalConstraintList.Skip
            yield False
            yield LogicalConstraintList.End
            yield m.y

        self.assertEqual(len(m.p), 4)
        self.assertExpressionsStructurallyEqual(m.p[0].expr, m.x)
        self.assertExpressionsStructurallyEqual(m.p[1].expr, m.x.implies(m.y))
        self.assertExpressionsStructurallyEqual(m.p[2].expr, BooleanConstant(True))
        self.assertExpressionsStructurallyEqual(m.p[3].expr, BooleanConstant(False))

        with self.assertRaisesRegex(
            ValueError, "LogicalConstraintList does not accept the 'expr' keyword"
        ):
            m.q = LogicalConstraintList(expr=[m.x])

        with LoggingIntercept(level=logging.DEBUG, module='pyomo.core.base') as LOG:
            m.r = LogicalConstraintList()
        self.assertEqual(LOG.getvalue(), 'Constructing logical constraint list r\n')

        with LoggingIntercept(level=logging.DEBUG, module='pyomo.core.base') as LOG:
            m.r.construct()
        self.assertEqual(LOG.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
