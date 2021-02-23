#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# Unit Tests for expression generation
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
from pyutilib.th import nottest

from pyomo.environ import ConcreteModel, RangeSet, Param, Var, Expression, ExternalFunction, VarList, sum_product, inequality, quicksum, sin, tanh
from pyomo.core.expr.numvalue import nonpyomo_leaf_types
from pyomo.core.expr.numeric_expr import (
    SumExpression, ProductExpression, 
    MonomialTermExpression, LinearExpression)
from pyomo.core.expr.visitor import (
    FixedExpressionError, NonConstantExpressionError,
    StreamBasedExpressionVisitor, ExpressionReplacementVisitor,
    evaluate_expression, expression_to_string, replace_expressions,
    sizeof_expression,
    identify_variables, identify_components, identify_mutable_parameters,
)
from pyomo.core.base.param import _ParamData, SimpleParam
from pyomo.core.expr.template_expr import IndexTemplate
from pyomo.core.expr.expr_errors import TemplateExpressionError
from pyomo.common.log import LoggingIntercept
from six import StringIO


class TestExpressionUtilities(unittest.TestCase):

    def test_identify_vars_numeric(self):
        #
        # There are no variables in a constant expression
        #
        self.assertEqual( list(identify_variables(5)), [] )

    def test_identify_vars_params(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Param(initialize=1)
        m.b = Param(m.I, initialize=1, mutable=True)
        #
        # There are no variables in expressions with only parameters
        #
        self.assertEqual( list(identify_variables(m.a)), [] )
        self.assertEqual( list(identify_variables(m.b[1])), [] )
        self.assertEqual( list(identify_variables(m.a+m.b[1])), [] )
        self.assertEqual( list(identify_variables(m.a**m.b[1])), [] )
        self.assertEqual( list(identify_variables(
            m.a**m.b[1] + m.b[2])), [] )
        self.assertEqual( list(identify_variables(
            m.a**m.b[1] + m.b[2]*m.b[3]*m.b[2])), [] )

    def test_identify_duplicate_vars(self):
        #
        # Identify variables when there are duplicates
        #
        m = ConcreteModel()
        m.a = Var(initialize=1)

        #self.assertEqual( list(identify_variables(2*m.a+2*m.a, allow_duplicates=True)),
        #                  [ m.a, m.a ] )
        self.assertEqual( list(identify_variables(2*m.a+2*m.a)),
                          [ m.a ] )

    def test_identify_vars_expr(self):
        #
        # Identify variables when there are duplicates
        #
        m = ConcreteModel()
        m.a = Var(initialize=1)
        m.b = Var(initialize=2)
        m.e = Expression(expr=3*m.a)
        m.E = Expression([0,1], initialize={0:3*m.a, 1:4*m.b})

        self.assertEqual( list(identify_variables(m.b+m.e)), [ m.b, m.a ] )
        self.assertEqual( list(identify_variables(m.E[0])), [ m.a ] )
        self.assertEqual( list(identify_variables(m.E[1])), [ m.b ] )

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
        self.assertEqual( list(identify_variables(m.a)), [m.a] )
        self.assertEqual( list(identify_variables(m.b[1])), [m.b[1]] )
        self.assertEqual( list(identify_variables(m.a+m.b[1])),
                          [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_variables(m.a**m.b[1])),
                          [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_variables(m.a**m.b[1] + m.b[2])),
                          [ m.a, m.b[1], m.b[2] ] )
        self.assertEqual( list(identify_variables(
            m.a**m.b[1] + m.b[2]*m.b[3]*m.b[2])),
                          [ m.a, m.b[1], m.b[2], m.b[3] ] )
        self.assertEqual( list(identify_variables(
            m.a**m.b[1] + m.b[2]/m.b[3]*m.b[2])),
                          [ m.a, m.b[1], m.b[2], m.b[3] ] )
        #
        # Identify variables in the arguments to functions
        #
        self.assertEqual( list(identify_variables(
            m.x(m.a, 'string_param', 1, [])*m.b[1] )),
                          [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_variables(
            m.x(m.p, 'string_param', 1, [])*m.b[1] )),
                          [ m.b[1] ] )
        self.assertEqual( list(identify_variables(
            tanh(m.a)*m.b[1] )), [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_variables(
            abs(m.a)*m.b[1] )), [ m.a, m.b[1] ] )
        #
        # Check logic for allowing duplicates
        #
        self.assertEqual( list(identify_variables(m.a**m.a + m.a)),
                          [ m.a ] )
        #self.assertEqual( list(identify_variables(m.a**m.a + m.a, allow_duplicates=True)),
        #                  [ m.a, m.a, m.a,  ] )

    def test_identify_vars_linear_expression(self):
        m = ConcreteModel()
        m.x = Var()
        expr = quicksum([m.x, m.x], linear=True)
        self.assertEqual(list(identify_variables(
            expr, include_fixed=False)), [m.x])



class TestIdentifyParams(unittest.TestCase):

    def test_identify_params_numeric(self):
        #
        # There are no parameters in a constant expression
        #
        self.assertEqual( list(identify_mutable_parameters(5)), [] )

    def test_identify_mutable_parameters(self):
        m = ConcreteModel()
        m.I = RangeSet(3)
        m.a = Var(initialize=1)
        m.b = Var(m.I, initialize=1)
        #
        # There are no variables in expressions with only vars
        #
        self.assertEqual( list(identify_mutable_parameters(m.a)), [] )
        self.assertEqual( list(identify_mutable_parameters(m.b[1])), [] )
        self.assertEqual( list(identify_mutable_parameters(m.a+m.b[1])), [] )
        self.assertEqual( list(identify_mutable_parameters(m.a**m.b[1])), [] )
        self.assertEqual( list(identify_mutable_parameters(
            m.a**m.b[1] + m.b[2])), [] )
        self.assertEqual( list(identify_mutable_parameters(
            m.a**m.b[1] + m.b[2]*m.b[3]*m.b[2])), [] )

    def test_identify_duplicate_params(self):
        #
        # Identify mutable params when there are duplicates
        #
        m = ConcreteModel()
        m.a = Param(initialize=1, mutable=True)

        self.assertEqual( list(identify_mutable_parameters(2*m.a+2*m.a)),
                          [ m.a ] )

    def test_identify_mutable_parameters_expr(self):
        #
        # Identify variables when there are duplicates
        #
        m = ConcreteModel()
        m.a = Param(initialize=1, mutable=True)
        m.b = Param(initialize=2, mutable=True)
        m.e = Expression(expr=3*m.a)
        m.E = Expression([0,1], initialize={0:3*m.a, 1:4*m.b})

        self.assertEqual( list(identify_mutable_parameters(m.b+m.e)), [ m.b, m.a ] )
        self.assertEqual( list(identify_mutable_parameters(m.E[0])), [ m.a ] )
        self.assertEqual( list(identify_mutable_parameters(m.E[1])), [ m.b ] )

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
        self.assertEqual( list(identify_mutable_parameters(m.a)), [m.a] )
        self.assertEqual( list(identify_mutable_parameters(m.b[1])), [m.b[1]] )
        self.assertEqual( list(identify_mutable_parameters(m.a+m.b[1])),
                          [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_mutable_parameters(m.a**m.b[1])),
                          [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_mutable_parameters(m.a**m.b[1] + m.b[2])),
                          [ m.a, m.b[1], m.b[2] ] )
        self.assertEqual( list(identify_mutable_parameters(
            m.a**m.b[1] + m.b[2]*m.b[3]*m.b[2])),
                          [ m.a, m.b[1], m.b[2], m.b[3] ] )
        self.assertEqual( list(identify_mutable_parameters(
            m.a**m.b[1] + m.b[2]/m.b[3]*m.b[2])),
                          [ m.a, m.b[1], m.b[2], m.b[3] ] )
        #
        # Identify variables in the arguments to functions
        #
        self.assertEqual( list(identify_mutable_parameters(
            m.x(m.a, 'string_param', 1, [])*m.b[1] )),
                          [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_mutable_parameters(
            m.x(m.p, 'string_param', 1, [])*m.b[1] )),
                          [ m.b[1] ] )
        self.assertEqual( list(identify_mutable_parameters(
            tanh(m.a)*m.b[1] )), [ m.a, m.b[1] ] )
        self.assertEqual( list(identify_mutable_parameters(
            abs(m.a)*m.b[1] )), [ m.a, m.b[1] ] )
        #
        # Check logic for allowing duplicates
        #
        self.assertEqual( list(identify_mutable_parameters(m.a**m.a + m.a)),
                          [ m.a ] )


#
# Replace all variables with new variables from a varlist
#
class ReplacementWalkerTest1(ExpressionReplacementVisitor):

    def __init__(self, model):
        ExpressionReplacementVisitor.__init__(self)
        self.model = model

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types or\
           not node.is_potentially_variable():
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

        e = sin(M.x) + M.x*M.y + 3
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("sin(x) + x*y + 3", str(e))
        self.assertEqual("sin(w[1]) + w[1]*w[2] + 3", str(f))

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()

        e = M.x
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("x", str(e))
        self.assertEqual("w[1]", str(f))

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x*M.y + 3 <= 0
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("sin(x) + x*y + 3  <=  0.0", str(e))
        self.assertEqual("sin(w[1]) + w[1]*w[2] + 3  <=  0.0", str(f))

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = inequality(0, sin(M.x) + M.x*M.y + 3, 1)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("0  <=  sin(x) + x*y + 3  <=  1", str(e))
        self.assertEqual("0  <=  sin(w[1]) + w[1]*w[2] + 3  <=  1", str(f))

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)

        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("z[0]*x[0] + z[1]*x[1] + z[2]*x[2]", str(e))
        self.assertEqual("z[0]*w[1] + z[1]*w[2] + z[2]*w[3]", str(f))

        e = 2*sum_product(M.z, M.x)
        walker = ReplacementWalkerTest1(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("2*(z[0]*x[0] + z[1]*x[1] + z[2]*x[2])", str(e))
        self.assertEqual("2*(z[0]*w[4] + z[1]*w[5] + z[2]*w[6])", str(f))

    def test_replace_expressions_with_monomial_term(self):
        M = ConcreteModel()
        M.x = Var()
        e = 2.0*M.x
        substitution_map = {id(M.x): 3.0*M.x}
        new_e = replace_expressions(e, substitution_map=substitution_map)
        self.assertEqual('6.0*x', str(new_e))
        # See comment about this test in ExpressionReplacementVisitor
        # old code would print '2.0*3.0*x'

    def test_identify_components(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = Var()

        e = sin(M.x) + M.x*M.w + 3
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

        e = sin(M.x) + M.x*M.w + 3
        v = list(str(v) for v in identify_variables(e))
        self.assertEqual(v, ['x', 'w'])
        v = list(str(v) for v in identify_variables(e, include_fixed=False))
        self.assertEqual(v, ['x'])

    def test_expression_to_string(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = Var()

        e = sin(M.x) + M.x*M.w + 3
        self.assertEqual("sin(x) + x*w + 3", expression_to_string(e))
        M.w = 2
        M.w.fixed = True
        self.assertEqual("sin(x) + x*2 + 3", expression_to_string(e, compute_values=True))

    def test_expression_component_to_string(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.e = Expression(expr=m.x*m.y)
        m.f = Expression(expr=m.e)

        e = m.x + m.f*m.y
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
        if node.__class__ in nonpyomo_leaf_types or\
           not node.is_potentially_variable():
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

        e = sin(M.x) + M.x*M.y + 3
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("sin(x) + x*y + 3", str(e))
        self.assertEqual("sin(2*w[1]) + 2*w[1]*(2*w[2]) + 3", str(f))

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()

        e = M.x
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("x", str(e))
        self.assertEqual("2*w[1]", str(f))

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x*M.y + 3 <= 0
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("sin(x) + x*y + 3  <=  0.0", str(e))
        self.assertEqual("sin(2*w[1]) + 2*w[1]*(2*w[2]) + 3  <=  0.0", str(f))

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Var()
        M.y = Var()
        M.w = VarList()

        e = inequality(0, sin(M.x) + M.x*M.y + 3, 1)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("0  <=  sin(x) + x*y + 3  <=  1", str(e))
        self.assertEqual("0  <=  sin(2*w[1]) + 2*w[1]*(2*w[2]) + 3  <=  1", str(f))

    def test_replacement_walker5(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)

        e = M.z*M.x
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(e.__class__ is MonomialTermExpression)
        self.assertTrue(f.__class__ is ProductExpression)
        self.assertEqual("z*x", str(e))
        self.assertEqual("z*(2*w[1])", str(f))

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)

        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("z[0]*x[0] + z[1]*x[1] + z[2]*x[2]", str(e))
        self.assertEqual("z[0]*(2*w[1]) + z[1]*(2*w[2]) + z[2]*(2*w[3])", str(f))

        e = 2*sum_product(M.z, M.x)
        walker = ReplacementWalkerTest2(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("2*(z[0]*x[0] + z[1]*x[1] + z[2]*x[2])", str(e))
        self.assertEqual("2*(z[0]*(2*w[4]) + z[1]*(2*w[5]) + z[2]*(2*w[6]))", str(f))

#
# Replace all mutable parameters with variables
#
class ReplacementWalkerTest3(ExpressionReplacementVisitor):

    def __init__(self, model):
        ExpressionReplacementVisitor.__init__(self)
        self.model = model

    def visiting_potential_leaf(self, node):
        if node.__class__ in (_ParamData, SimpleParam):
            if id(node) in self.substitute:
                return True, self.substitute[id(node)]
            self.substitute[id(node)] = 2*self.model.w.add()
            return True, self.substitute[id(node)]

        if node.__class__ in nonpyomo_leaf_types or \
            node.is_constant() or \
            node.is_variable_type():
            return True, node

        return False, None


class WalkerTests3(unittest.TestCase):

    def test_replacement_walker1(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x*M.y + 3
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("sin(x) + x*y + 3", str(e))
        self.assertEqual("sin(2*w[1]) + 2*w[1]*y + 3", str(f))

    def test_replacement_walker2(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.w = VarList()

        e = M.x
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("x", str(e))
        self.assertEqual("2*w[1]", str(f))

    def test_replacement_walker3(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()

        e = sin(M.x) + M.x*M.y + 3 <= 0
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("sin(x) + x*y + 3  <=  0.0", str(e))
        self.assertEqual("sin(2*w[1]) + 2*w[1]*y + 3  <=  0.0", str(f))

    def test_replacement_walker4(self):
        M = ConcreteModel()
        M.x = Param(mutable=True)
        M.y = Var()
        M.w = VarList()

        e = inequality(0, sin(M.x) + M.x*M.y + 3, 1)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("0  <=  sin(x) + x*y + 3  <=  1", str(e))
        self.assertEqual("0  <=  sin(2*w[1]) + 2*w[1]*y + 3  <=  1", str(f))

    def test_replacement_walker5(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)

        e = M.z*M.x
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(e.__class__ is MonomialTermExpression)
        self.assertTrue(f.__class__ is ProductExpression)
        self.assertTrue(f.arg(0).is_potentially_variable())
        self.assertEqual("z*x", str(e))
        self.assertEqual("2*w[1]*x", str(f))

    def test_replacement_walker6(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)

        e = (M.z*2)*3
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(not e.is_potentially_variable())
        self.assertTrue(f.is_potentially_variable())
        self.assertEqual("z*2*3", str(e))
        self.assertEqual("2*w[1]*2*3", str(f))

    def test_replacement_walker7(self):
        M = ConcreteModel()
        M.x = Var()
        M.w = VarList()
        M.z = Param(mutable=True)
        M.e = Expression(expr=M.z*2)

        e = M.x*M.e
        self.assertTrue(e.arg(1).is_potentially_variable())
        self.assertTrue(not e.arg(1).arg(0).is_potentially_variable())
        self.assertEqual("x*(z*2)", str(e))
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertTrue(e.__class__ is ProductExpression)
        self.assertTrue(f.__class__ is ProductExpression)
        self.assertEqual(id(e), id(f))
        self.assertTrue(f.arg(1).is_potentially_variable())
        self.assertTrue(f.arg(1).arg(0).is_potentially_variable())
        self.assertEqual("x*(2*w[1]*2)", str(f))

    def test_replacement_walker0(self):
        M = ConcreteModel()
        M.x = Var(range(3))
        M.w = VarList()
        M.z = Param(range(3), mutable=True)

        e = sum_product(M.z, M.x)
        self.assertIs(type(e), LinearExpression)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("z[0]*x[0] + z[1]*x[1] + z[2]*x[2]", str(e))
        self.assertEqual("2*w[1]*x[0] + 2*w[2]*x[1] + 2*w[3]*x[2]", str(f))

        e = 2*sum_product(M.z, M.x)
        walker = ReplacementWalkerTest3(M)
        f = walker.dfs_postorder_stack(e)
        self.assertEqual("2*(z[0]*x[0] + z[1]*x[1] + z[2]*x[2])", str(e))
        self.assertEqual("2*(2*w[4]*x[0] + 2*w[5]*x[1] + 2*w[6]*x[2])", str(f))


#
# Replace all mutable parameters with variables
#
class ReplacementWalker_ReplaceInternal(ExpressionReplacementVisitor):

    def __init__(self):
        ExpressionReplacementVisitor.__init__(self)

    def visit(self, node, values):
        if type(node) == ProductExpression:
            return sum(values)
        else:
            return node


class WalkerTests_ReplaceInternal(unittest.TestCase):

    def test_no_replacement(self):
        m = ConcreteModel()
        m.x = Param(mutable=True)
        m.y = Var([1,2,3])

        e = sum(m.y[i] for i in m.y) == 0
        f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
        self.assertEqual("y[1] + y[2] + y[3]  ==  0.0", str(e))
        self.assertEqual("y[1] + y[2] + y[3]  ==  0.0", str(f))
        self.assertIs(e, f)

    def test_replace(self):
        m = ConcreteModel()
        m.x = Param(mutable=True)
        m.y = Var([1,2,3])

        e = m.y[1]*m.y[2] + m.y[2]*m.y[3]  ==  0
        f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
        self.assertEqual("y[1]*y[2] + y[2]*y[3]  ==  0.0", str(e))
        self.assertEqual("y[1] + y[2] + (y[2] + y[3])  ==  0.0", str(f))
        self.assertIs(type(f.arg(0)), SumExpression)
        self.assertEqual(f.arg(0).nargs(), 2)
        self.assertIs(type(f.arg(0).arg(0)), SumExpression)
        self.assertEqual(f.arg(0).arg(0).nargs(), 2)
        self.assertIs(type(f.arg(0).arg(1)), SumExpression)
        self.assertEqual(f.arg(0).arg(1).nargs(), 2)

    def test_replace_nested(self):
        m = ConcreteModel()
        m.x = Param(mutable=True)
        m.y = Var([1,2,3])

        e = m.y[1]*m.y[2]*m.y[2]*m.y[3]  ==  0
        f = ReplacementWalker_ReplaceInternal().dfs_postorder_stack(e)
        self.assertEqual("y[1]*y[2]*y[2]*y[3]  ==  0.0", str(e))
        self.assertEqual("y[1] + y[2] + y[2] + y[3]  ==  0.0", str(f))
        self.assertIs(type(f.arg(0)), SumExpression)
        self.assertEqual(f.arg(0).nargs(), 4)


class TestStreamBasedExpressionVisitor(unittest.TestCase):
    def setUp(self):
        self.m = m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        self.e = m.x**2 + m.y + m.z*(m.x+m.y)

    def test_bad_args(self):
        with self.assertRaisesRegexp(
                RuntimeError, "Unrecognized keyword arguments: {'foo': None}"):
            StreamBasedExpressionVisitor(foo=None)

    def test_default(self):
        walker = StreamBasedExpressionVisitor()
        ans = walker.walk_expression(self.e)
        ref = [
            [[],[]],
            [],
            [[],[[],[]],]
        ]
        self.assertEqual(ans, ref)

    def test_beforeChild(self):
        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types \
               or not child.is_expression_type():
                return False, [child]
        walker = StreamBasedExpressionVisitor(beforeChild=before)
        ans = walker.walk_expression(self.e)
        m = self.m
        ref = [
            [[m.x], [2]],
            [m.y],
            [[m.z], [[m.x], [m.y]]]
        ]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(m.x)
        ref = []
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(2)
        ref = []
        self.assertEqual(str(ans), str(ref))

    def test_old_beforeChild(self):
        def before(node, child):
            if type(child) in nonpyomo_leaf_types \
               or not child.is_expression_type():
                return False, [child]
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = StreamBasedExpressionVisitor(beforeChild=before)
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the beforeChild() "
            "method", os.getvalue().replace('\n',' '))

        ans = walker.walk_expression(self.e)
        m = self.m
        ref = [
            [[m.x], [2]],
            [m.y],
            [[m.z], [[m.x], [m.y]]]
        ]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(m.x)
        ref = []
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(2)
        ref = []
        self.assertEqual(str(ans), str(ref))

    def test_reduce_in_accept(self):
        def enter(node):
            return None, 1
        def accept(node, data, child_result, child_idx):
            return data + child_result
        walker = StreamBasedExpressionVisitor(
            enterNode=enter, acceptChildResult=accept)
        # 4 operators, 6 leaf nodes
        self.assertEquals(walker.walk_expression(self.e), 10)

    def test_sizeof_expression(self):
        self.assertEquals(sizeof_expression(self.e), 10)

    def test_enterNode(self):
        # This is an alternative way to implement the beforeChild test:
        def enter(node):
            if type(node) in nonpyomo_leaf_types \
               or not node.is_expression_type():
                return (), [node]
            return node.args, []
        walker = StreamBasedExpressionVisitor(
            enterNode=enter)
        m = self.m

        ans = walker.walk_expression(self.e)
        ref = [
            [[m.x], [2]],
            [m.y],
            [[m.z], [[m.x], [m.y]]]
        ]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_enterNode_noLeafList(self):
        # This is an alternative way to implement the beforeChild test:
        def enter(node):
            if type(node) in nonpyomo_leaf_types \
               or not node.is_expression_type():
                return (), node
            return node.args, []
        walker = StreamBasedExpressionVisitor(
            enterNode=enter)
        m = self.m

        ans = walker.walk_expression(self.e)
        ref = [
            [m.x, 2],
            m.y,
            [m.z, [m.x, m.y]]
        ]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(m.x)
        ref = m.x
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(2)
        ref = 2
        self.assertEqual(str(ans), str(ref))

    def test_enterNode_withFinalize(self):
        # This is an alternative way to implement the beforeChild test:
        def enter(node):
            if type(node) in nonpyomo_leaf_types \
               or not node.is_expression_type():
                return (), node
            return node.args, []
        def finalize(result):
            if type(result) is list:
                return result
            else:
                return[result]
        walker = StreamBasedExpressionVisitor(
            enterNode=enter, finalizeResult=finalize)
        m = self.m

        ans = walker.walk_expression(self.e)
        ref = [
            [m.x, 2],
            m.y,
            [m.z, [m.x, m.y]]
        ]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(2)
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

        ans = walker.walk_expression(self.e)
        ref = [
            [[m.x], [2]],
            [m.y],
            [[m.z], [[m.x], [m.y]]]
        ]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(m.x)
        ref = [m.x]
        self.assertEqual(str(ans), str(ref))

        ans = walker.walk_expression(2)
        ref = [2]
        self.assertEqual(str(ans), str(ref))

    def test_beforeChild_acceptChildResult_afterChild(self):
        counts = [0,0,0]
        def before(node, child, child_idx):
            counts[0] += 1
            if type(child) in nonpyomo_leaf_types \
               or not child.is_expression_type():
                return False, None
        def accept(node, data, child_result, child_idx):
            counts[1] += 1
        def after(node, child, child_idx):
            counts[2] += 1
        walker = StreamBasedExpressionVisitor(
            beforeChild=before, acceptChildResult=accept, afterChild=after)
        ans = walker.walk_expression(self.e)
        m = self.m
        self.assertEqual(ans, None)
        self.assertEquals(counts, [9,9,9])

    def test_OLD_beforeChild_acceptChildResult_afterChild(self):
        counts = [0,0,0]
        def before(node, child):
            counts[0] += 1
            if type(child) in nonpyomo_leaf_types \
               or not child.is_expression_type():
                return False, None
        def accept(node, data, child_result):
            counts[1] += 1
        def after(node, child):
            counts[2] += 1

        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = StreamBasedExpressionVisitor(
                beforeChild=before, acceptChildResult=accept, afterChild=after)
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "beforeChild() method", os.getvalue().replace('\n',' '))
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "acceptChildResult() method", os.getvalue().replace('\n',' '))
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "afterChild() method", os.getvalue().replace('\n',' '))

        ans = walker.walk_expression(self.e)
        m = self.m
        self.assertEqual(ans, None)
        self.assertEquals(counts, [9,9,9])

    def test_enterNode_acceptChildResult_beforeChild(self):
        ans = []
        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types \
               or not child.is_expression_type():
                return False, child
        def accept(node, data, child_result, child_idx):
            if data is not child_result:
                data.append(child_result)
            return data
        def enter(node):
            return node.args, ans
        walker = StreamBasedExpressionVisitor(
            enterNode=enter, beforeChild=before, acceptChildResult=accept)
        ans = walker.walk_expression(self.e)
        m = self.m
        ref = [m.x, 2, m.y, m.z, m.x, m.y]
        self.assertEqual(str(ans), str(ref))

    def test_finalize(self):
        ans = []
        def before(node, child, child_idx):
            if type(child) in nonpyomo_leaf_types \
               or not child.is_expression_type():
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
            enterNode=enter, beforeChild=before, acceptChildResult=accept,
            finalizeResult=finalize)
        ans = walker.walk_expression(self.e)
        self.assertEqual(ans, 6)

    def test_all_function_pointers(self):
        ans = []
        def name(x):
            if type(x) in nonpyomo_leaf_types:
                return str(x)
            else:
                return x.name
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
            enterNode=enter, exitNode=exit, beforeChild=before, 
            acceptChildResult=accept, afterChild=after, finalizeResult=finalize)
        self.assertIsNone( walker.walk_expression(self.e) )
        self.assertEqual("\n".join(ans),"""Enter sum
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
Finalize""")

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
            def enterNode(self, node):
                self.ans.append("Enter %s" % (name(node)))
            def exitNode(self, node, data):
                self.ans.append("Exit %s" % (name(node)))
            def beforeChild(self, node, child, child_idx):
                self.ans.append("Before %s (from %s)"
                                % (name(child), name(node)))
            def acceptChildResult(self, node, data, child_result, child_idx):
                self.ans.append("Accept into %s" % (name(node)))
            def afterChild(self, node, child, child_idx):
                self.ans.append("After %s (from %s)"
                                % (name(child), name(node)))
            def finalizeResult(self, result):
                self.ans.append("Finalize")
        walker = all_callbacks()
        self.assertIsNone( walker.walk_expression(self.e) )
        self.assertEqual("\n".join(walker.ans),"""Enter sum
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
Finalize""")

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
                self.ans.append("Before %s (from %s)"
                                % (name(child), name(node)))
            def acceptChildResult(self, node, data, child_result):
                self.ans.append("Accept into %s" % (name(node)))
            def afterChild(self, node, child):
                self.ans.append("After %s (from %s)"
                                % (name(child), name(node)))
            def finalizeResult(self, result):
                self.ans.append("Finalize")
        os = StringIO()
        with LoggingIntercept(os, 'pyomo'):
            walker = all_callbacks()
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "beforeChild() method", os.getvalue().replace('\n',' '))
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "acceptChildResult() method", os.getvalue().replace('\n',' '))
        self.assertIn(
            "Note that the API for the StreamBasedExpressionVisitor "
            "has changed to include the child index for the "
            "afterChild() method", os.getvalue().replace('\n',' '))

        self.assertIsNone( walker.walk_expression(self.e) )
        self.assertEqual("\n".join(walker.ans),"""Enter sum
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
Finalize""")


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
        self.assertRaises(NonConstantExpressionError, evaluate_expression, e, constant=True)

    def test_initialized_variable(self):
        m = ConcreteModel()
        m.p = Var(initialize=1)

        e = 1 + m.p
        self.assertTrue(2, evaluate_expression(e))
        self.assertRaises(NonConstantExpressionError, evaluate_expression, e, constant=True)

    def test_fixed_variable(self):
        m = ConcreteModel()
        m.p = Var(initialize=1)
        m.p.fixed = True

        e = 1 + m.p
        self.assertTrue(2, evaluate_expression(e))
        self.assertRaises(FixedExpressionError, evaluate_expression, e, constant=True)

    def test_template_expr(self):
        m = ConcreteModel()
        m.I = RangeSet(1,9)
        m.x = Var(m.I, initialize=lambda m,i: i+1)
        m.P = Param(m.I, initialize=lambda m,i: 10-i, mutable=True)
        t = IndexTemplate(m.I)

        e = m.x[t+m.P[t+1]] + 3
        self.assertRaises(TemplateExpressionError, evaluate_expression, e)
        self.assertRaises(TemplateExpressionError, evaluate_expression, e, constant=True)

if __name__ == "__main__":
    unittest.main()
