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
#
# Unit Tests for expression generation
#

import os
import platform
import sys

import pyomo.common.unittest as unittest

from pyomo.environ import (
    ConcreteModel,
    RangeSet,
    Set,
    Param,
    Var,
    Expression,
    ExternalFunction,
    VarList,
    sum_product,
    inequality,
    quicksum,
    sin,
    tanh,
    value,
)
from pyomo.core.expr.numvalue import native_types, nonpyomo_leaf_types, NumericConstant
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    ProductExpression,
    MonomialTermExpression,
    LinearExpression,
    NPV_SumExpression,
    NPV_ProductExpression,
    NegationExpression,
    NPV_NegationExpression,
    PowExpression,
    NPV_PowExpression,
    MaxExpression,
    NPV_MaxExpression,
    MinExpression,
    NPV_MinExpression,
    DivisionExpression,
    NPV_DivisionExpression,
    UnaryFunctionExpression,
    NPV_UnaryFunctionExpression,
    AbsExpression,
    NPV_AbsExpression,
)
from pyomo.core.expr.visitor import (
    FixedExpressionError,
    NonConstantExpressionError,
    StreamBasedExpressionVisitor,
    ExpressionReplacementVisitor,
    evaluate_expression,
    expression_to_string,
    replace_expressions,
    sizeof_expression,
    identify_variables,
    identify_components,
    identify_mutable_parameters,
    RECURSION_LIMIT,
    get_stack_depth,
)
from pyomo.core.base.param import _ParamData, ScalarParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.common.collections import ComponentSet
from pyomo.common.errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from io import StringIO
from pyomo.core.expr.compare import assertExpressionsEqual


class TestExpressionUtilities(unittest.TestCase):
    def test_identify_vars_numeric(self):
        #
        # There are no variables in a constant expression
        #
        self.assertEqual(list(identify_variables(5)), [])

    def test_identify_vars_params(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Param(initialize=1)
        m.b = Param(m.I, initialize=1, mutable=True)
        #
        # There are no variables in expressions with only parameters
        #
        self.assertEqual(list(identify_variables(m.a)), [])
        self.assertEqual(list(identify_variables(m.b[1])), [])
        self.assertEqual(list(identify_variables(m.a + m.b[1])), [])
        self.assertEqual(list(identify_variables(m.a ** m.b[1])), [])
        self.assertEqual(list(identify_variables(m.a ** m.b[1] + m.b[2])), [])
        self.assertEqual(
            list(identify_variables(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])), []
        )

    def test_identify_duplicate_vars(self):
        #
        # Identify variables when there are duplicates
        #
        m = ConcreteModel()
        m.a = Var(initialize=1)

        # self.assertEqual( list(identify_variables(2*m.a+2*m.a, allow_duplicates=True)),
        #                  [ m.a, m.a ] )
        self.assertEqual(list(identify_variables(2 * m.a + 2 * m.a)), [m.a])

    def test_identify_vars_expr(self):
        #
        # Identify variables in named expressions
        #
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3 * m.a)
        m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})

        self.assertEqual(list(identify_variables(m.b + m.e)), [m.b, m.a])
        self.assertEqual(list(identify_variables(m.E[0])), [m.a])
        self.assertEqual(list(identify_variables(m.E[1])), [m.b])

    def test_identify_vars_vars(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Var(initialize=1)
        m.b = Var(m.I, initialize=1)
        m.p = Param(initialize=1, mutable=True)
        m.x = ExternalFunction(library='foo.so', function='bar')
        #
        # Identify variables in various algebraic expressions
        #
        self.assertEqual(list(identify_variables(m.a)), [m.a])
        self.assertEqual(list(identify_variables(m.b[1])), [m.b[1]])
        self.assertEqual(list(identify_variables(m.a + m.b[1])), [m.a, m.b[1]])
        self.assertEqual(list(identify_variables(m.a ** m.b[1])), [m.a, m.b[1]])
        self.assertEqual(
            list(identify_variables(m.a ** m.b[1] + m.b[2])), [m.b[2], m.a, m.b[1]]
        )
        self.assertEqual(
            list(identify_variables(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])),
            [m.a, m.b[1], m.b[2], m.b[3]],
        )
        self.assertEqual(
            list(identify_variables(m.a ** m.b[1] + m.b[2] / m.b[3] * m.b[2])),
            [m.a, m.b[1], m.b[2], m.b[3]],
        )
        #
        # Identify variables in the arguments to functions
        #
        self.assertEqual(
            list(identify_variables(m.x(m.a, 'string_param', 1, []) * m.b[1])),
            [m.b[1], m.a],
        )
        self.assertEqual(
            list(identify_variables(m.x(m.p, 'string_param', 1, []) * m.b[1])), [m.b[1]]
        )
        self.assertEqual(list(identify_variables(tanh(m.a) * m.b[1])), [m.b[1], m.a])
        self.assertEqual(list(identify_variables(abs(m.a) * m.b[1])), [m.b[1], m.a])
        #
        # Check logic for allowing duplicates
        #
        self.assertEqual(list(identify_variables(m.a**m.a + m.a)), [m.a])
        # self.assertEqual( list(identify_variables(m.a**m.a + m.a, allow_duplicates=True)),
        #                  [ m.a, m.a, m.a,  ] )

    def test_identify_vars_linear_expression(self):
        m = ConcreteModel()
        m.x = Var()
        expr = quicksum([m.x, m.x], linear=True)
        self.assertEqual(list(identify_variables(expr, include_fixed=False)), [m.x])


class TestIdentifyParams(unittest.TestCase):
    def test_identify_params_numeric(self):
        #
        # There are no parameters in a constant expression
        #
        self.assertEqual(list(identify_mutable_parameters(5)), [])

    def test_identify_mutable_parameters(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Var(initialize=1)
        m.b = Var(m.I, initialize=1)
        #
        # There are no parameters in expressions with only vars
        #
        self.assertEqual(list(identify_mutable_parameters(m.a)), [])
        self.assertEqual(list(identify_mutable_parameters(m.b[1])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a + m.b[1])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1])), [])
        self.assertEqual(list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2])), [])
        self.assertEqual(
            list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])),
            [],
        )

    def test_identify_mutable_parameters_constants(self):
        #
        # ScalarParams and NumericConstants are not recognized
        #
        m = ConcreteModel()
        m.x = Var(initialize=1)
        m.x.fix()
        m.p = Param(initialize=2, mutable=False)
        m.p_m = Param(initialize=3, mutable=True)
        e1 = m.x + m.p + NumericConstant(5)
        self.assertEqual(list(identify_mutable_parameters(e1)), [])

        e2 = 5 * m.x + NumericConstant(3) * m.p_m + m.p == 0
        mut_params = list(identify_mutable_parameters(e2))
        self.assertEqual(len(mut_params), 1)
        self.assertIs(mut_params[0], m.p_m)

    def test_identify_duplicate_params(self):
        #
        # Identify mutable params when there are duplicates
        #
        m = ConcreteModel()
        m.a = Param(initialize=1, mutable=True)

        self.assertEqual(list(identify_mutable_parameters(2 * m.a + 2 * m.a)), [m.a])

    def test_identify_mutable_parameters_expr(self):
        #
        # Identify mutable params in named expressions
        #
        m = ConcreteModel()
        m.a = Param(initialize=1, mutable=True)
        m.b = Param(initialize=2, mutable=True)
        m.e = Expression(expr=3 * m.a)
        m.E = Expression([0, 1], initialize={0: 3 * m.a, 1: 4 * m.b})

        self.assertEqual(list(identify_mutable_parameters(m.b + m.e)), [m.b, m.a])
        self.assertEqual(list(identify_mutable_parameters(m.E[0])), [m.a])
        self.assertEqual(list(identify_mutable_parameters(m.E[1])), [m.b])

    def test_identify_mutable_parameters_logical_expr(self):
        #
        # Identify mutable params in logical expressions
        #
        m = ConcreteModel()
        m.a = Param(initialize=0, mutable=True)
        expr = m.a + 1 == 0
        param_set = ComponentSet(identify_mutable_parameters(expr))
        self.assertEqual(len(param_set), 1)
        self.assertIn(m.a, param_set)

    def test_identify_mutable_parameters_params(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Param(initialize=1, mutable=True)
        m.b = Param(m.I, initialize=1, mutable=True)
        m.p = Var(initialize=1)
        m.x = ExternalFunction(library='foo.so', function='bar')
        #
        # Identify variables in various algebraic expressions
        #
        self.assertEqual(list(identify_mutable_parameters(m.a)), [m.a])
        self.assertEqual(list(identify_mutable_parameters(m.b[1])), [m.b[1]])
        self.assertEqual(list(identify_mutable_parameters(m.a + m.b[1])), [m.a, m.b[1]])
        self.assertEqual(
            list(identify_mutable_parameters(m.a ** m.b[1])), [m.a, m.b[1]]
        )
        self.assertEqual(
            list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2])),
            [m.b[2], m.a, m.b[1]],
        )
        self.assertEqual(
            list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] * m.b[3] * m.b[2])),
            [m.a, m.b[1], m.b[2], m.b[3]],
        )
        self.assertEqual(
            list(identify_mutable_parameters(m.a ** m.b[1] + m.b[2] / m.b[3] * m.b[2])),
            [m.a, m.b[1], m.b[2], m.b[3]],
        )
        #
        # Identify variables in the arguments to functions
        #
        self.assertEqual(
            list(identify_mutable_parameters(m.x(m.a, 'string_param', 1, []) * m.b[1])),
            [m.b[1], m.a],
        )
        self.assertEqual(
            list(identify_mutable_parameters(m.x(m.p, 'string_param', 1, []) * m.b[1])),
            [m.b[1]],
        )
        self.assertEqual(
            list(identify_mutable_parameters(tanh(m.a) * m.b[1])), [m.b[1], m.a]
        )
        self.assertEqual(
            list(identify_mutable_parameters(abs(m.a) * m.b[1])), [m.b[1], m.a]
        )
        #
        # Check logic for allowing duplicates
        #
        self.assertEqual(list(identify_mutable_parameters(m.a**m.a + m.a)), [m.a])


#
# Replace all variables with new variables from a varlist
#
class ReplacementWalkerTest1(ExpressionReplacementVisitor):
    def __init__(self, model):
        ExpressionReplacementVisitor.__init__(self)
        self.model = model

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types or not node.is_potentially_variable():
            return True, node
        if node.is_variable_type():
            if id(node) in self.substitute:
                return True, self.substitute[id(node)]
            self.substitute[id(node)] = self.model.w.add()
            return True, self.substitute[id(node)]
        return False, None


class WalkerTests(unittest.TestCase):
    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x * M.y + 3
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3, e)
        assertExpressionsEqual(self, sin(M.w[1]) + M.w[1] * M.w[2] + 3, f)

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()

        e = M.x
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, M.x, e)
        assertExpressionsEqual(self, M.w[1], f)

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x * M.y + 3 <= 0
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
        assertExpressionsEqual(self, sin(M.w[1]) + M.w[1] * M.w[2] + 3 <= 0, f)

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = inequality(0, sin(M.x) + M.x * M.y + 3, 1)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, inequality(0, sin(M.x) + M.x * M.y + 3, 1), e)
        assertExpressionsEqual(
            self, inequality(0, sin(M.w[1]) + M.w[1] * M.w[2] + 3, 1), f
        )

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)

        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(
            self,
            LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.x.values()],
            ),
            e,
        )
        assertExpressionsEqual(
            self,
            LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.w.values()],
            ),
            f,
        )

        del M.w
        del M.w_index
        M.w = VarList()
        e = 2 * sum_product(M.z, M.x)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(
            self,
            2
            * LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.x.values()],
            ),
            e,
        )
        assertExpressionsEqual(
            self,
            2
            * LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.w.values()],
            ),
            f,
        )

    def test_replacement_linear_expression_with_constant(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        e = LinearExpression(linear_coefs=[2], linear_vars=[m.x])
        e += m.y
        sub_map = dict()
        sub_map[id(m.x)] = 5
        e2 = replace_expressions(e, sub_map)
        assertExpressionsEqual(
            self, e2, LinearExpression([10, MonomialTermExpression((1, m.y))])
        )

        e = LinearExpression(linear_coefs=[2, 3], linear_vars=[m.x, m.y])
        sub_map = dict()
        sub_map[id(m.x)] = 5
        e2 = replace_expressions(e, sub_map)
        assertExpressionsEqual(
            self, e2, LinearExpression([10, MonomialTermExpression((3, m.y))])
        )

    def test_replacement_linear_expression_with_nonlinear(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        e = LinearExpression(linear_coefs=[2, 3], linear_vars=[m.x, m.y])
        sub_map = dict()
        sub_map[id(m.x)] = m.x**2
        e2 = replace_expressions(e, sub_map)
        assertExpressionsEqual(self, e2, SumExpression([2 * m.x**2, 3 * m.y]))

    def test_replace_expressions_with_monomial_term(self):
        M = ConcreteModel()
        M.x = Var()
        e = 2.0 * M.x
        substitution_map = {id(M.x): 3.0 * M.x}
        new_e = replace_expressions(e, substitution_map=substitution_map)
        self.assertEqual('6.0*x', str(new_e))
        # See comment about this test in ExpressionReplacementVisitor
        # old code would print '2.0*3.0*x'

    def test_identify_components(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = Var()

        e = sin(M.x) + M.x * M.w + 3
        v = list(str(v) for v in identify_components(e, set([M.x.__class__])))
        self.assertEqual(v, ['x', 'w'])
        v = list(str(v) for v in identify_components(e, [M.x.__class__]))
        self.assertEqual(v, ['x', 'w'])

    def test_identify_variables(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = Var()
        M.w = 2
        M.w.fixed = True

        e = sin(M.x) + M.x * M.w + 3
        v = list(str(v) for v in identify_variables(e))
        self.assertEqual(v, ['x', 'w'])
        v = list(str(v) for v in identify_variables(e, include_fixed=False))
        self.assertEqual(v, ['x'])

    def test_expression_to_string(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = Var()

        e = sin(M.x) + M.x * M.w + 3
        self.assertEqual("sin(x) + x*w + 3", expression_to_string(e))
        M.w = 2
        M.w.fixed = True
        self.assertEqual(
            "sin(x) + x*2 + 3", expression_to_string(e, compute_values=True)
        )

    def test_expression_component_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=m.x * m.y)
        m.f = Expression(expr=m.e)

        e = m.x + m.f * m.y
        self.assertEqual("x + ((x*y))*y", str(e))
        self.assertEqual("x + ((x*y))*y", expression_to_string(e))


#
# Replace all variables with a product expression
#
class ReplacementWalkerTest2(ExpressionReplacementVisitor):
    def __init__(self, model):
        ExpressionReplacementVisitor.__init__(self)
        self.model = model

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types or not node.is_potentially_variable():
            return True, node

        if node.is_variable_type():
            if id(node) in self.substitute:
                return True, self.substitute[id(node)]
            self.substitute[id(node)] = 2 * self.model.w.add()
            return True, self.substitute[id(node)]
        return False, None


class WalkerTests2(unittest.TestCase):
    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x * M.y + 3
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * (2 * M.w[2]) + 3, f)

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()

        e = M.x
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, M.x, e)
        assertExpressionsEqual(self, 2 * M.w[1], f)

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x * M.y + 3 <= 0
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
        assertExpressionsEqual(
            self, sin(2 * M.w[1]) + 2 * M.w[1] * (2 * M.w[2]) + 3 <= 0, f
        )

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = inequality(0, sin(M.x) + M.x * M.y + 3, 1)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, inequality(0, sin(M.x) + M.x * M.y + 3, 1), e)
        assertExpressionsEqual(
            self, inequality(0, sin(2 * M.w[1]) + 2 * M.w[1] * (2 * M.w[2]) + 3, 1), f
        )

    def test_replacement_walker5(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)

        e = M.z * M.x
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, e, MonomialTermExpression((M.z, M.x)))
        assertExpressionsEqual(
            self, f, MonomialTermExpression((NPV_ProductExpression((M.z, 2)), M.w[1]))
        )

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)

        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(
            self,
            e,
            LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.x.values()],
            ),
        )
        assertExpressionsEqual(
            self,
            f,
            LinearExpression(
                [
                    MonomialTermExpression(
                        (NPV_ProductExpression((M.z[0], 2)), M.w[1])
                    ),
                    MonomialTermExpression(
                        (NPV_ProductExpression((M.z[1], 2)), M.w[2])
                    ),
                    MonomialTermExpression(
                        (NPV_ProductExpression((M.z[2], 2)), M.w[3])
                    ),
                ]
            ),
        )

        e = 2 * sum_product(M.z, M.x)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(
            self,
            2
            * LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.x.values()],
            ),
            e,
        )
        assertExpressionsEqual(
            self,
            f,
            ProductExpression(
                (
                    2,
                    LinearExpression(
                        [
                            MonomialTermExpression(
                                (NPV_ProductExpression((M.z[0], 2)), M.w[4])
                            ),
                            MonomialTermExpression(
                                (NPV_ProductExpression((M.z[1], 2)), M.w[5])
                            ),
                            MonomialTermExpression(
                                (NPV_ProductExpression((M.z[2], 2)), M.w[6])
                            ),
                        ]
                    ),
                )
            ),
        )


#
# Replace all mutable parameters with variables
#
class ReplacementWalkerTest3(ExpressionReplacementVisitor):
    def __init__(self, model):
        super().__init__(remove_named_expressions=False)
        self.model = model

    def visiting_potential_leaf(self, node):
        if node.__class__ in (_ParamData, ScalarParam):
            if id(node) in self.substitute:
                return True, self.substitute[id(node)]
            self.substitute[id(node)] = 2 * self.model.w.add()
            return True, self.substitute[id(node)]

        if (
            node.__class__ in nonpyomo_leaf_types
            or node.is_constant()
            or node.is_variable_type()
        ):
            return True, node

        return False, None


class WalkerTests3(unittest.TestCase):
    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x * M.y + 3
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3, f)

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.w = VarList()

        e = M.x
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, M.x, e)
        assertExpressionsEqual(self, 2 * M.w[1], f)

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x * M.y + 3 <= 0
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, sin(M.x) + M.x * M.y + 3 <= 0, e)
        assertExpressionsEqual(self, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3 <= 0, f)

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()

        e = inequality(0, sin(M.x) + M.x * M.y + 3, 1)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(self, inequality(0, sin(M.x) + M.x * M.y + 3, 1), e)
        assertExpressionsEqual(
            self, inequality(0, sin(2 * M.w[1]) + 2 * M.w[1] * M.y + 3, 1), f
        )

    def test_replacement_walker5(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)

        e = M.z * M.x
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertIs(e.__class__, MonomialTermExpression)
        self.assertIs(f.__class__, ProductExpression)
        self.assertTrue(f.arg(0).is_potentially_variable())
        assertExpressionsEqual(self, M.z * M.x, e)
        assertExpressionsEqual(self, 2 * M.w[1] * M.x, f)

    def test_replacement_walker6(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)

        e = (M.z * 2) * 3
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(not e.is_potentially_variable())
        self.assertTrue(f.is_potentially_variable())
        assertExpressionsEqual(self, M.z * 2 * 3, e)
        assertExpressionsEqual(
            self, ProductExpression([ProductExpression([2 * M.w[1], 2]), 3]), f
        )

    def test_replacement_walker7(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)
        M.e = Expression(expr=M.z * 2)

        e = M.x * M.e
        self.assertTrue(e.arg(1).is_potentially_variable())
        self.assertTrue(not e.arg(1).arg(0).is_potentially_variable())
        assertExpressionsEqual(
            self,
            ProductExpression([M.x, (NPV_ProductExpression([M.z, 2]))]),
            e,
            include_named_exprs=False,
        )
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(e.__class__ is ProductExpression)
        self.assertTrue(f.__class__ is ProductExpression)
        self.assertEqual(id(e), id(f))
        self.assertTrue(f.arg(1).is_potentially_variable())
        self.assertTrue(f.arg(1).arg(0).is_potentially_variable())
        assertExpressionsEqual(
            self, M.x * ProductExpression([2 * M.w[1], 2]), f, include_named_exprs=False
        )

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)

        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(
            self,
            LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.x.values()],
            ),
            e,
        )
        assertExpressionsEqual(
            self, 2 * M.w[1] * M.x[0] + 2 * M.w[2] * M.x[1] + 2 * M.w[3] * M.x[2], f
        )

        e = 2 * sum_product(M.z, M.x)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        assertExpressionsEqual(
            self,
            2
            * LinearExpression(
                linear_coefs=[i for i in M.z.values()],
                linear_vars=[i for i in M.x.values()],
            ),
            e,
        )
        assertExpressionsEqual(
            self,
            2 * (2 * M.w[4] * M.x[0] + 2 * M.w[5] * M.x[1] + 2 * M.w[6] * M.x[2]),
            f,
        )


#
# Replace all mutable parameters with variables
#
class ReplacementWalker_ReplaceInternal(ExpressionReplacementVisitor):
    def exitNode(self, node, data):
        if type(node) == ProductExpression:
            return sum(data[1])
        else:
            return super().exitNode(node, data)


class WalkerTests_ReplaceInternal(unittest.TestCase):
    def test_no_replacement(self):
        m = ConcreteModel()
        m.x = Param(mutable=True)
        m.y = Var([1, 2, 3])

        e = sum(m.y[i] for i in m.y) == 0
        f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
        assertExpressionsEqual(self, m.y[1] + m.y[2] + m.y[3] == 0, e)
        assertExpressionsEqual(self, m.y[1] + m.y[2] + m.y[3] == 0, f)
        self.assertIs(e, f)

    def test_replace(self):
        m = ConcreteModel()
        m.x = Param(mutable=True)
        m.y = Var([1, 2, 3])

        e = m.y[1] * m.y[2] + m.y[2] * m.y[3] == 0
        f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
        assertExpressionsEqual(self, m.y[1] * m.y[2] + m.y[2] * m.y[3] == 0, e)
        assertExpressionsEqual(
            self,
            SumExpression(
                [
                    LinearExpression(
                        [
                            MonomialTermExpression((1, m.y[1])),
                            MonomialTermExpression((1, m.y[2])),
                        ]
                    ),
                    LinearExpression(
                        [
                            MonomialTermExpression((1, m.y[2])),
                            MonomialTermExpression((1, m.y[3])),
                        ]
                    ),
                ]
            )
            == 0,
            f,
        )

    def test_replace_nested(self):
        m = ConcreteModel()
        m.x = Param(mutable=True)
        m.y = Var([1, 2, 3])

        e = m.y[1] * m.y[2] * m.y[2] * m.y[3] == 0
        f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
        assertExpressionsEqual(self, m.y[1] * m.y[2] * m.y[2] * m.y[3] == 0, e)
        assertExpressionsEqual(self, m.y[1] + m.y[2] + m.y[2] + m.y[3] == 0, f)
        self.assertIs(type(f.arg(0)), LinearExpression)
        self.assertEqual(f.arg(0).nargs(), 4)


class TestReplacementWithNPV(unittest.TestCase):
    def test_npv_sum(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()

        e1 = m.p1 + 2
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, m.p2 + 2)
        assertExpressionsEqual(
            self, e3, LinearExpression([MonomialTermExpression((1, m.x)), 2])
        )

    def test_npv_negation(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()

        e1 = -m.p1
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, -m.p2)
        assertExpressionsEqual(self, e3, NegationExpression([m.x]))

    def test_npv_pow(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()

        e1 = m.p1**3
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, m.p2**3)
        assertExpressionsEqual(self, e3, m.x**3)

    def test_npv_product(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()

        e1 = m.p1 * 3
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, m.p2 * 3)
        assertExpressionsEqual(self, e3, ProductExpression([m.x, 3]))

    def test_npv_div(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()

        e1 = m.p1 / 3
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, m.p2 / 3)
        assertExpressionsEqual(self, e3, DivisionExpression((m.x, 3)))

    def test_npv_unary(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var(initialize=0)

        e1 = sin(m.p1)
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, sin(m.p2))
        assertExpressionsEqual(self, e3, sin(m.x))

    def test_npv_abs(self):
        m = ConcreteModel()
        m.p1 = Param(mutable=True)
        m.p2 = Param(mutable=True)
        m.x = Var()

        e1 = abs(m.p1)
        e2 = replace_expressions(e1, {id(m.p1): m.p2})
        e3 = replace_expressions(e1, {id(m.p1): m.x})

        assertExpressionsEqual(self, e2, abs(m.p2))
        assertExpressionsEqual(self, e3, abs(m.x))


class BaseStreamBasedVisitorTests(object):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        # Note: we do not use the operator overloading to generate the
        # expression so that the structure is constant even when we make
        # adjustments to the expression generators
        self.e = SumExpression(
            [
                PowExpression((m.x, 2)),
                m.y,
                ProductExpression((m.z, SumExpression([m.x, m.y]))),
            ]
        )

    def test_bad_args(self):
        with self.assertRaisesRegex(
            RuntimeError, "Unrecognized keyword arguments: {'foo': None}"
        ):
            StreamBasedExpressionVisitor(foo=None)

    def test_default(self):
        walker = StreamBasedExpressionVisitor()
        ans = self.walk(walker, self.e)
        ref = [[[], []], [], [[], [[], []]]]
        self.assertEqual(ans, ref)

    def test_beforeChild(self):
        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, [child]

        walker = StreamBasedExpressionVisitor(beforeChild=before)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = []
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = []
        self.assertEqual(str(ans), str(ref))

    def test_initializeWalker_beforeChild(self):
        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, child

        def initialize(expr):
            ans = before(None, expr, 0)
            if ans is None:
                return True, expr
            else:
                return ans

        walker = StreamBasedExpressionVisitor(
            beforeChild=before, initializeWalker=initialize
        )
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [[m.x, 2], m.y, [m.z, [m.x, m.y]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = m.x
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = 2
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_exitNode(self):
        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, [child]

        def exit(node, data):
            if hasattr(node, 'getname'):
                data.insert(0, node.getname())
            else:
                data.insert(0, str(node))
            return data

        walker = StreamBasedExpressionVisitor(beforeChild=before, exitNode=exit)
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [
            'sum',
            ['pow', [m.x], [2]],
            [m.y],
            ['prod', [m.z], ['sum', [m.x], [m.y]]],
        ]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = ['x']
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = ['2']
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_enterNode_exitNode(self):
        i = [0]

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, [child]

        def enter(node):
            i[0] += 1
            return None, [i[0]]

        def exit(node, data):
            if hasattr(node, 'getname'):
                data.insert(0, node.getname())
            else:
                data.insert(0, str(node))
            return data

        walker = StreamBasedExpressionVisitor(
            beforeChild=before, enterNode=enter, exitNode=exit
        )
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [
            'sum',
            1,
            ['pow', 2, [m.x], [2]],
            [m.y],
            ['prod', 3, [m.z], ['sum', 4, [m.x], [m.y]]],
        ]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = ['x', 5]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = ['2', 6]
        self.assertEqual(str(ans), str(ref))

    def test_old_beforeChild(self):
        def before(node, child):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, [child]

        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = StreamBasedExpressionVisitor(beforeChild=before)
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the beforeChild() "
            "method",
            os.getvalue().replace('\n', ' '),
        )

        ans = self.walk(walker, self.e)
        m = self.m
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = []
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = []
        self.assertEqual(str(ans), str(ref))

    def test_reduce_in_accept(self):
        def enter(node):
            return None, 1

        def accept(node, data, child_result, child_idx):
            return data + child_result

        walker = StreamBasedExpressionVisitor(enterNode=enter, acceptChildResult=accept)
        # 4 operators, 6 leaf nodes
        self.assertEqual(self.walk(walker, self.e), 10)

    def test_sizeof_expression(self):
        self.assertEqual(sizeof_expression(self.e), 10)

    def test_enterNode(self):
        # This is an alternative way to implement the beforeChild test:
        def enter(node):
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                return (), [node]
            return node.args, []

        walker = StreamBasedExpressionVisitor(enterNode=enter)
        m = self.m

        ans = self.walk(walker, self.e)
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_enterNode_noLeafList(self):
        # This is an alternative way to implement the beforeChild test:
        def enter(node):
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                return (), node
            return node.args, []

        walker = StreamBasedExpressionVisitor(enterNode=enter)
        m = self.m

        ans = self.walk(walker, self.e)
        ref = [[m.x, 2], m.y, [m.z, [m.x, m.y]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = m.x
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = 2
        self.assertEqual(str(ans), str(ref))

    def test_enterNode_withFinalize(self):
        # This is an alternative way to implement the beforeChild test:
        def enter(node):
            if type(node) in nonpyomo_leaf_types or not node.is_expression_type():
                return (), node
            return node.args, []

        def finalize(result):
            if type(result) is list:
                return result
            else:
                return [result]

        walker = StreamBasedExpressionVisitor(enterNode=enter, finalizeResult=finalize)
        m = self.m

        ans = self.walk(walker, self.e)
        ref = [[m.x, 2], m.y, [m.z, [m.x, m.y]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_exitNode(self):
        # This is an alternative way to implement the beforeChild test:
        def exit(node, data):
            if data:
                return data
            else:
                return [node]

        walker = StreamBasedExpressionVisitor(exitNode=exit)
        m = self.m

        ans = self.walk(walker, self.e)
        ref = [[[m.x], [2]], [m.y], [[m.z], [[m.x], [m.y]]]]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))

        ans = self.walk(walker, 2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_acceptChildResult_afterChild(self):
        counts = [0, 0, 0]

        def before(node, child, child_idx):
            counts[0] += 1
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, None

        def accept(node, data, child_result, child_idx):
            counts[1] += 1

        def after(node, child, child_idx):
            counts[2] += 1

        walker = StreamBasedExpressionVisitor(
            beforeChild=before, acceptChildResult=accept, afterChild=after
        )
        ans = self.walk(walker, self.e)
        m = self.m
        self.assertEqual(ans, None)
        self.assertEqual(counts, [9, 9, 9])

    def test_OLD_beforeChild_acceptChildResult_afterChild(self):
        counts = [0, 0, 0]

        def before(node, child):
            counts[0] += 1
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, None

        def accept(node, data, child_result):
            counts[1] += 1

        def after(node, child):
            counts[2] += 1

        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = StreamBasedExpressionVisitor(
                beforeChild=before, acceptChildResult=accept, afterChild=after
            )
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "beforeChild() method",
            os.getvalue().replace('\n', ' '),
        )
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "acceptChildResult() method",
            os.getvalue().replace('\n', ' '),
        )
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "afterChild() method",
            os.getvalue().replace('\n', ' '),
        )

        ans = self.walk(walker, self.e)
        m = self.m
        self.assertEqual(ans, None)
        self.assertEqual(counts, [9, 9, 9])

    def test_enterNode_acceptChildResult_beforeChild(self):
        ans = []

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, child

        def accept(node, data, child_result, child_idx):
            if data is not child_result:
                data.append(child_result)
            return data

        def enter(node):
            return node.args, ans

        walker = StreamBasedExpressionVisitor(
            enterNode=enter, beforeChild=before, acceptChildResult=accept
        )
        ans = self.walk(walker, self.e)
        m = self.m
        ref = [m.x, 2, m.y, m.z, m.x, m.y]
        self.assertEqual(str(ans), str(ref))

    def test_finalize(self):
        ans = []

        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types or not child.is_expression_type():
                return False, child

        def accept(node, data, child_result, child_idx):
            if data is not child_result:
                data.append(child_result)
            return data

        def enter(node):
            return node.args, ans

        def finalize(result):
            return len(result)

        walker = StreamBasedExpressionVisitor(
            enterNode=enter,
            beforeChild=before,
            acceptChildResult=accept,
            finalizeResult=finalize,
        )
        ans = self.walk(walker, self.e)
        self.assertEqual(ans, 6)

    def test_all_function_pointers(self):
        ans = []

        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name

        def initialize(expr):
            ans.append("Initialize")
            return True, None

        def enter(node):
            ans.append("Enter %s" % (name(node)))

        def exit(node, data):
            ans.append("Exit %s" % (name(node)))

        def before(node, child, child_idx):
            ans.append("Before %s (from %s)" % (name(child), name(node)))

        def accept(node, data, child_result, child_idx):
            ans.append("Accept into %s" % (name(node)))

        def after(node, child, child_idx):
            ans.append("After %s (from %s)" % (name(child), name(node)))

        def finalize(result):
            ans.append("Finalize")

        walker = StreamBasedExpressionVisitor(
            initializeWalker=initialize,
            enterNode=enter,
            exitNode=exit,
            beforeChild=before,
            acceptChildResult=accept,
            afterChild=after,
            finalizeResult=finalize,
        )
        self.assertIsNone(self.walk(walker, self.e))
        self.assertEqual(
            "\n".join(ans),
            """Initialize
Enter sum
Before pow (from sum)
Enter pow
Before x (from pow)
Enter x
Exit x
Accept into pow
After x (from pow)
Before 2 (from pow)
Enter 2
Exit 2
Accept into pow
After 2 (from pow)
Exit pow
Accept into sum
After pow (from sum)
Before y (from sum)
Enter y
Exit y
Accept into sum
After y (from sum)
Before prod (from sum)
Enter prod
Before z (from prod)
Enter z
Exit z
Accept into prod
After z (from prod)
Before sum (from prod)
Enter sum
Before x (from sum)
Enter x
Exit x
Accept into sum
After x (from sum)
Before y (from sum)
Enter y
Exit y
Accept into sum
After y (from sum)
Exit sum
Accept into prod
After sum (from prod)
Exit prod
Accept into sum
After prod (from sum)
Exit sum
Finalize""",
        )

    def test_all_derived_class(self):
        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name

        class all_callbacks(StreamBasedExpressionVisitor):
            def __init__(self):
                self.ans = []
                super(all_callbacks, self).__init__()

            def initializeWalker(self, expr):
                self.ans.append("Initialize")
                return True, None

            def enterNode(self, node):
                self.ans.append("Enter %s" % (name(node)))

            def exitNode(self, node, data):
                self.ans.append("Exit %s" % (name(node)))

            def beforeChild(self, node, child, child_idx):
                self.ans.append("Before %s (from %s)" % (name(child), name(node)))

            def acceptChildResult(self, node, data, child_result, child_idx):
                self.ans.append("Accept into %s" % (name(node)))

            def afterChild(self, node, child, child_idx):
                self.ans.append("After %s (from %s)" % (name(child), name(node)))

            def finalizeResult(self, result):
                self.ans.append("Finalize")

        walker = all_callbacks()
        self.assertIsNone(self.walk(walker, self.e))
        self.assertEqual(
            "\n".join(walker.ans),
            """Initialize
Enter sum
Before pow (from sum)
Enter pow
Before x (from pow)
Enter x
Exit x
Accept into pow
After x (from pow)
Before 2 (from pow)
Enter 2
Exit 2
Accept into pow
After 2 (from pow)
Exit pow
Accept into sum
After pow (from sum)
Before y (from sum)
Enter y
Exit y
Accept into sum
After y (from sum)
Before prod (from sum)
Enter prod
Before z (from prod)
Enter z
Exit z
Accept into prod
After z (from prod)
Before sum (from prod)
Enter sum
Before x (from sum)
Enter x
Exit x
Accept into sum
After x (from sum)
Before y (from sum)
Enter y
Exit y
Accept into sum
After y (from sum)
Exit sum
Accept into prod
After sum (from prod)
Exit prod
Accept into sum
After prod (from sum)
Exit sum
Finalize""",
        )

    def test_all_derived_class_oldAPI(self):
        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name

        class all_callbacks(StreamBasedExpressionVisitor):
            def __init__(self):
                self.ans = []
                super(all_callbacks, self).__init__()

            def enterNode(self, node):
                self.ans.append("Enter %s" % (name(node)))

            def exitNode(self, node, data):
                self.ans.append("Exit %s" % (name(node)))

            def beforeChild(self, node, child):
                self.ans.append("Before %s (from %s)" % (name(child), name(node)))

            def acceptChildResult(self, node, data, child_result):
                self.ans.append("Accept into %s" % (name(node)))

            def afterChild(self, node, child):
                self.ans.append("After %s (from %s)" % (name(child), name(node)))

            def finalizeResult(self, result):
                self.ans.append("Finalize")

        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = all_callbacks()
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "beforeChild() method",
            os.getvalue().replace('\n', ' '),
        )
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "acceptChildResult() method",
            os.getvalue().replace('\n', ' '),
        )
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "afterChild() method",
            os.getvalue().replace('\n', ' '),
        )

        self.assertIsNone(self.walk(walker, self.e))
        self.assertEqual(
            "\n".join(walker.ans),
            """Enter sum
Before pow (from sum)
Enter pow
Before x (from pow)
Enter x
Exit x
Accept into pow
After x (from pow)
Before 2 (from pow)
Enter 2
Exit 2
Accept into pow
After 2 (from pow)
Exit pow
Accept into sum
After pow (from sum)
Before y (from sum)
Enter y
Exit y
Accept into sum
After y (from sum)
Before prod (from sum)
Enter prod
Before z (from prod)
Enter z
Exit z
Accept into prod
After z (from prod)
Before sum (from prod)
Enter sum
Before x (from sum)
Enter x
Exit x
Accept into sum
After x (from sum)
Before y (from sum)
Enter y
Exit y
Accept into sum
After y (from sum)
Exit sum
Accept into prod
After sum (from prod)
Exit prod
Accept into sum
After prod (from sum)
Exit sum
Finalize""",
        )


class TestStreamBasedExpressionVisitor_Recursive(
    BaseStreamBasedVisitorTests, unittest.TestCase
):
    def walk(self, walker, expr):
        return walker.walk_expression(expr)


class TestStreamBasedExpressionVisitor_NonRecursive(
    BaseStreamBasedVisitorTests, unittest.TestCase
):
    def walk(self, walker, expr):
        return walker.walk_expression_nonrecursive(expr)


def fill_stack(n, fcn, *args):
    if n:
        return fill_stack(n - 1, fcn, *args)
    else:
        return fcn(*args)


class TestStreamBasedExpressionVisitor_Deep(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.x = Var()
        m.I = Set(initialize=range(2 * RECURSION_LIMIT))

        def _rule(m, i):
            if i:
                return m.e[i - 1]
            else:
                return m.x

        m.e = Expression(m.I, rule=_rule)

    def evaluate_bx(self):
        def before(node, child, idx):
            if type(child) in native_types or not child.is_expression_type():
                return False, value(child)
            return True, None

        def exit(node, data):
            return data[0] + 1

        return StreamBasedExpressionVisitor(beforeChild=before, exitNode=exit)

    def evaluate_bex(self):
        def before(node, child, idx):
            if type(child) in native_types or not child.is_expression_type():
                return False, value(child)
            return True, None

        def enter(node):
            return None, []

        def exit(node, data):
            return data[0] + 1

        return StreamBasedExpressionVisitor(
            beforeChild=before, enterNode=enter, exitNode=exit
        )

    def evaluate_abex(self):
        def before(node, child, idx):
            if type(child) in native_types or not child.is_expression_type():
                return False, value(child)
            return True, None

        def enter(node):
            return None, 0

        def accept(node, data, child_result, child_idx):
            return data + child_result

        def exit(node, data):
            return data + 1

        return StreamBasedExpressionVisitor(
            beforeChild=before, acceptChildResult=accept, enterNode=enter, exitNode=exit
        )

    def run_walker(self, walker):
        m = self.m
        m.x = 10
        self.assertEqual(
            2 * RECURSION_LIMIT + 10,
            walker.walk_expression(m.e[2 * RECURSION_LIMIT - 1]),
        )
        self.assertEqual(
            2 * RECURSION_LIMIT + 10,
            walker.walk_expression_nonrecursive(m.e[2 * RECURSION_LIMIT - 1]),
        )

        # This is a "magic parameter" that quantifies the overhead
        # needed by the system to convert the recursive walker to a
        # nonrecursive one.
        #
        # Note: this needs to be 13 if pytest is run as a script, and 14
        # if pytest is run as "python -m".  We will use 14, and then add
        # 2 (instead of 1) to generate the recursion error.  Note that
        # the stack handling is different on GHA, and we need to fill an
        # additional frame (for a total of 3) to trigger the recursion
        # error.
        #
        TESTING_OVERHEAD = 14
        warn_msg = "Unexpected RecursionError walking an expression tree.\n"

        if platform.python_implementation() == 'PyPy':
            # We have not yet determined how to trigger the
            # RecursionError on PyPy
            cases = [(0, "")]
        elif os.environ.get('GITHUB_ACTIONS', '') and sys.platform.startswith('win'):
            # The test for handling RecursionError appears to fail
            # inexplicably on GHA/Windows under pytest: the
            # RecursionError that is supposed to be raised is not
            # raised, and instead the system actually dies on stack
            # overflow error
            cases = []
        else:
            # 3 sufficed through Python 3.10, but appeared to need to be
            # raised to 5 for Python 3.11 builds (3.11.2), and again to
            # 10 for Python 3.12 builds (3.12.0)
            cases = [(0, ""), (10, warn_msg)]

        head_room = sys.getrecursionlimit() - get_stack_depth()
        for n, msg in cases:
            with LoggingIntercept() as LOG:
                self.assertEqual(
                    2 * RECURSION_LIMIT + 10,
                    fill_stack(
                        head_room - RECURSION_LIMIT - TESTING_OVERHEAD + n,
                        walker.walk_expression,
                        m.e[2 * RECURSION_LIMIT - 1],
                    ),
                )
            self.assertEqual(msg, LOG.getvalue())

    def test_evaluate_bx(self):
        return self.run_walker(self.evaluate_bx())

    def test_evaluate_bex(self):
        return self.run_walker(self.evaluate_bex())

    def test_evaluate_abex(self):
        return self.run_walker(self.evaluate_abex())


class TestEvaluateExpression(unittest.TestCase):
    def test_constant(self):
        m = ConcreteModel()
        m.p = Param(initialize=1)

        e = 1 + m.p
        self.assertEqual(2, evaluate_expression(e))
        self.assertEqual(2, evaluate_expression(e, constant=True))

    def test_uninitialized_constant(self):
        m = ConcreteModel()
        m.p = Param(mutable=True)

        e = 1 + m.p
        self.assertRaises(ValueError, evaluate_expression, e)
        self.assertRaises(FixedExpressionError, evaluate_expression, e, constant=True)

    def test_variable(self):
        m = ConcreteModel()
        m.p = Var()

        e = 1 + m.p
        self.assertRaises(ValueError, evaluate_expression, e)
        self.assertRaises(
            NonConstantExpressionError, evaluate_expression, e, constant=True
        )

    def test_initialized_variable(self):
        m = ConcreteModel()
        m.p = Var(initialize=1)

        e = 1 + m.p
        self.assertTrue(2, evaluate_expression(e))
        self.assertRaises(
            NonConstantExpressionError, evaluate_expression, e, constant=True
        )

    def test_fixed_variable(self):
        m = ConcreteModel()
        m.p = Var(initialize=1)
        m.p.fixed = True

        e = 1 + m.p
        self.assertTrue(2, evaluate_expression(e))
        self.assertRaises(FixedExpressionError, evaluate_expression, e, constant=True)

    def test_template_expr(self):
        m = ConcreteModel()
        m.I = RangeSet(1, 9)
        m.x = Var(m.I, initialize=lambda m, i: i + 1)
        m.P = Param(m.I, initialize=lambda m, i: 10 - i, mutable=True)
        t = IndexTemplate(m.I)

        e = m.x[t + m.P[t + 1]] + 3
        self.assertRaises(TemplateExpressionError, evaluate_expression, e)
        self.assertRaises(
            TemplateExpressionError, evaluate_expression, e, constant=True
        )


if __name__ == "__main__":
    unittest.main()
