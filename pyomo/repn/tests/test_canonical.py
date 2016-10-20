#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Test the canonical expressions
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

from pyomo.core.base.expr import Expr_if
from pyomo.repn import *
from pyomo.environ import *

from six import iteritems
from six.moves import range

class frozendict(dict):
    __slots__ = ('_hash',)
    def __hash__(self):
        rval = getattr(self, '_hash', None)
        if rval is None:
            rval = self._hash = hash(frozenset(iteritems(self)))
        return rval

# a silly utility to facilitate comparison of tuples where we don't care about ordering
def linear_repn_to_dict(repn):
    result = {}
    if repn.variables is not None:
        for i in range(len(repn.variables)):
            result[id(repn.variables[i])] = repn.linear[i]
    if repn.constant != None:
        result[None] = repn.constant
    return result

class Test(unittest.TestCase):

    #def setUp(self):
        #
        # Create Model
        #
        #self.plugin = SimplePreprocessor()
        #self.plugin.deactivate_action("compute_canonical_repn")

    def tearDown(self):
        if os.path.exists("unknown.lp"):
            os.unlink("unknown.lp")
        pyutilib.services.TempfileManager.clear_tempfiles()
        #self.plugin.activate_action("compute_canonical_repn")

    def test_abstract_linear_expression(self):
        m = AbstractModel()
        m.A = RangeSet(1,3)
        def p_init(model, i):
            return 2*i
        m.p = Param(m.A, initialize=p_init)
        m.x = Var(m.A, bounds=(-1,1))
        def obj_rule(model):
            return summation(model.p, model.x)
        m.obj = Objective(rule=obj_rule)
        i = m.create_instance()

        rep = generate_canonical_repn(i.obj[None].expr)
        # rep should only have variables and linear terms
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant == None)
        self.assertTrue(rep.variables != None)
        # rep.variables should have the 3 variables...
        baseline = { id(i.x[1]) : 2,
                     id(i.x[2]) : 4,
                     id(i.x[3]) : 6 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_concrete_linear_expression(self):
        m = ConcreteModel()
        m.A = RangeSet(1,3)
        def p_init(model, i):
            return 2*i
        m.p = Param(m.A, initialize=p_init)
        m.x = Var(m.A, bounds=(-1,1))
        m.obj = Objective(expr=summation(m.p, m.x))
        rep = generate_canonical_repn(m.obj[None].expr)
        # rep should only have variables and linear terms
        self.assertTrue(rep.variables != None)
        self.assertTrue(rep.linear != None)
        # rep.variables should have the 3 variables...
        baseline = { id(m.x[1]) : 2,
                     id(m.x[2]) : 4,
                     id(m.x[3]) : 6 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_linear_expression(self):
        I = range(3)
        x = [Var(bounds=(-1,1)) for i in I]
        expr = sum(2*(i+1)*x[i] for i in I)
        rep = generate_canonical_repn(expr)
        # rep should only have variables and linear terms
        self.assertTrue(rep.variables != None)
        self.assertTrue(rep.linear != None)
        # rep.variables should have the 3 variables...
        baseline = { id(x[0]) : 2,
                     id(x[1]) : 4,
                     id(x[2]) : 6 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_complex_linear_expression(self):
        I = range(3)
        x = [Var(bounds=(-1,1)) for i in I]
        expr = 2*x[1] + 3*x[1] + 4*(x[2]+x[1])
        rep = generate_canonical_repn(expr)
        # rep should only have variables and linear components
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.variables != None)
        # rep.variables should have the 2 of the 3 variables...
        baseline = { id(x[1]) : 9.0,
                     id(x[2]) : 4.0 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))


    def test_reversed_complex_linear_expression(self):
        I = range(3)
        x = [Var(bounds=(-1,1)) for i in I]
        expr = 4*(x[2]+x[1]) + 2*x[1] + 3*x[1]
        rep = generate_canonical_repn(expr)
        # rep should only have variables and linear terms
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.variables != None)
        # rep.variables should have the 2 of the 3 variables...
        baseline = { id(x[2]) : 4.0,
                     id(x[1]) : 9.0 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_linear_expression_with_constant(self):
        I = range(3)
        x = [Var(bounds=(-1,1)) for i in I]
        expr = 1.2 + 2*x[1] + 3*x[1]
        rep = generate_canonical_repn(expr)
        # rep should only have [-1,0,1]
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.variables != None)
        # rep should have the 1 of the 3 variables...
        baseline = { id(x[1]) : 5.0,
                     None : 1.2 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_complex_linear_expression_with_constant(self):
        I = range(3)
        x = [Var(bounds=(-1,1)) for i in I]
        expr =  2.0*(1.2 + 2*x[1] + 3*x[1]) + 3.0*(1.0+x[1])
        rep = generate_canonical_repn(expr)
        # rep should only have variables, a constant term, and linear terms
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.variables != None)
        # rep.variables should have the 1 of the 3 variables...
        baseline = { id(x[1]) : 13.0,
                     None : 5.4 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_polynomial_expression(self):
        I = range(4)
        x = [Var(bounds=(-1,1)) for i in I]
        expr = x[1]*(x[1]+x[2]) + x[2]*(x[1]+3.0*x[3]*x[3])
        rep = generate_canonical_repn(expr)
        # rep should only have [-1,2,3]
        self.assertEqual(len(rep), 3)
        self.assertTrue(2 in rep)
        self.assertTrue(3 in rep)
        self.assertTrue(-1 in rep)
        # rep[-1] should have the 3 of the 4 variables...
        self.assertEqual(rep[-1], { 0: x[1],
                                    1: x[2],
                                    2: x[3] })
        # check the expression encoding
        self.assertEqual(rep[2], {frozendict({0:2}):1.0,
                                  frozendict({0:1, 1:1}):2.0})
        self.assertEqual(rep[3], {frozendict({1:1, 2:2}):3.0})


    def test_polynomial_expression_with_fixed(self):
        I = range(4)
        x = [Var() for i in I]
        for v in x:
            v.construct()
        expr = x[1]*(x[1]+x[2]) + x[2]*(x[1]+3.0*x[3]*x[3])
        x[1].value = 5
        x[1].fixed = True
        rep = generate_canonical_repn(expr)
        # rep should only have [-1,0,1,3]
        self.assertEqual(len(rep), 4)
        self.assertTrue(0 in rep)
        self.assertTrue(1 in rep)
        self.assertTrue(3 in rep)
        self.assertTrue(-1 in rep)
        # rep[-1] should have the 2 of the 4 variables...
        self.assertEqual(rep[-1], { 0: x[2],
                                    1: x[3] })
        # check the expression encoding
        self.assertEqual(rep[0], {None: 25.0})
        self.assertEqual(rep[1], {0: 10.0})
        self.assertEqual(rep[3], {frozendict({0:1, 1:2}):3.0})


    def test_linear_expression_with_constant_division(self):
        m = ConcreteModel()
        m.A = RangeSet(1,3)
        def p_init(model, i):
            return 2*i
        m.p = Param(m.A, initialize=p_init)
        m.x = Var(m.A, bounds=(-1,1))
        expr = summation(m.p, m.x)/2.0
        rep = generate_canonical_repn(expr)

        # rep should only have only variables and linear terms
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant == None)
        self.assertTrue(rep.variables != None)

        baseline = { id(m.x[1]) : 1.0,
                     id(m.x[2]) : 2.0,
                     id(m.x[3]) : 3.0 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_linear_expression_with_fixed_division(self):
        m = ConcreteModel()
        m.A = RangeSet(1,3)
        def p_init(model, i):
            return 2*i
        m.p = Param(m.A, initialize=p_init)
        m.x = Var(m.A, bounds=(-1,1))
        m.y = Var(initialize=2.0)
        m.y.fixed = True
        expr = summation(m.p, m.x)/m.y
        rep = generate_canonical_repn(expr)

        # rep should only have variables and linearm terms
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant == None)
        self.assertTrue(rep.variables != None)

        # rep.variables should have the 3 variables...
        baseline = { id(m.x[1]) : 1.0,
                     id(m.x[2]) : 2.0,
                     id(m.x[3]) : 3.0 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_linear_expression_with_complex_fixed_division(self):
        m = ConcreteModel()
        m.A = RangeSet(1,3)
        def p_init(model, i):
            return 2*i
        m.p = Param(m.A, initialize=p_init)
        m.x = Var(m.A, bounds=(-1,1))
        m.y = Var(initialize=1.0)
        m.y.fixed = True
        expr = summation(m.p, m.x)/(m.y+1)
        rep = generate_canonical_repn(expr)

        # rep should only have variable and a linear component
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant == None)
        self.assertTrue(rep.variables != None)

        # rep.variables should have the 3 variables...
        baseline = { id(m.x[1]) : 1.0,
                     id(m.x[2]) : 2.0,
                     id(m.x[3]) : 3.0 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_expr_rational_summation(self):
        m = ConcreteModel()
        m.A = RangeSet(1,3)
        def p_init(model, i):
            return 2*i
        m.p = Param(m.A, initialize=p_init)
        m.x = Var(m.A, bounds=(-1,1))
        m.y = Var(initialize=1.0)
        expr = summation(m.p, m.x)/(1+m.y)

        rep = generate_canonical_repn(expr)
        # rep should only have [-1,None]
        self.assertEqual(len(rep), 2)
        self.assertTrue(None in rep)
        self.assertTrue(-1 in rep)
        # rep[-1] should have the 4 variables...
        self.assertEqual(rep[-1], { 3: m.y,
                                    0: m.x[1],
                                    1: m.x[2],
                                    2: m.x[3] })
        # check the expression encoding
        self.assertIs(rep[None], expr)

    def test_expr_rational(self):
        m = ConcreteModel()
        m.A = RangeSet(1,4)
        m.x = Var(m.A, bounds=(-1,1))
        m.y = Var(initialize=1.0)
        expr = (1.25+m.x[1]+1/m.y) + (2.0+m.x[2])/m.y

        rep = generate_canonical_repn(expr)
        # rep should only have [-1,None]
        self.assertEqual(len(rep), 2)
        self.assertTrue(None in rep)
        self.assertTrue(-1 in rep)
        # rep[-1] should have the 3 variables...
        self.assertEqual(rep[-1], { 1: m.y,
                                    0: m.x[1],
                                    2: m.x[2] })
        # check the expression encoding
        self.assertIs(rep[None], expr)

    def test_expr_rational_fixed(self):
        m = ConcreteModel()
        m.A = RangeSet(1,4)
        m.x = Var(m.A, bounds=(-1,1))
        m.y = Var(initialize=2.0)
        m.y.fixed = True
        expr = (1.25+m.x[1]+1/m.y) + (2.0+m.x[2])/m.y

        rep = generate_canonical_repn(expr)

        # rep should only have variables, a constant, and linear terms
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.variables != None)

        # rep.variables should have the 2 of 3 variables...

        baseline = { id(m.x[1]) : 1.0,
                     id(m.x[2]) : 0.5,
                     None : 2.75 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_general_nonlinear(self):
        I = range(3)
        x = [Var() for i in I]
        expr = x[1] + 5*cos(x[2])

        rep = generate_canonical_repn(expr)
        # rep should only have [-1,None]
        self.assertEqual(len(rep), 2)
        self.assertTrue(None in rep)
        self.assertTrue(-1 in rep)
        # rep[-1] should have 2 variables...
        self.assertEqual(rep[-1], { 0: x[1],
                                    1: x[2] })
        # check the expression encoding
        self.assertIs(rep[None], expr)

    def test_general_nonlinear_fixed(self):
        I = range(3)
        x = [Var() for i in I]
        for v in x:
            v.construct()
        expr = x[1] + 5*cos(x[2])
        x[2].value = 0
        x[2].fixed = True

        rep = generate_canonical_repn(expr)
        # rep should only have variables, a constsant, and linear terms
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.variables != None)
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant != None)

        # rep.variables should have 1 variable...
        baseline = { id(x[1]) : 1.0,
                     None: 5.0 }
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_deterministic_var_labeling(self):
        m = ConcreteModel()
        m.x = Var(initialize=3.0)
        m.y = Var(initialize=2.0)
        exprA = m.x - m.y
        exprB = m.y - m.x

        # Nondeterministic form should not care which expression comes first
        # The nondeterministic form has been deprecated.
        #rep_A = generate_canonical_repn(exprA)
        #rep_B = generate_canonical_repn(exprB)
        #self.assertEqual( rep_A[-1], rep_B[-1] )

        # Deterministic form should care which expression comes first
        idMap_1 = {}
        rep_A = generate_canonical_repn(exprA, idMap_1)
        rep_B = generate_canonical_repn(exprB, idMap_1)
        self.assertEqual(set(map(lambda x: x.name, rep_A.variables)),
                         set(map(lambda x: x.name, rep_B.variables)))

        idMap_2 = {}
        rep_B = generate_canonical_repn(exprB, idMap_2)
        rep_A = generate_canonical_repn(exprA, idMap_2)
        self.assertEqual(set(map(lambda x: x.name, rep_A.variables)),
                         set(map(lambda x: x.name, rep_B.variables)))

        self.assertNotEqual( idMap_1, idMap_2 )

    def test_Expression_nonindexed(self):
        m = ConcreteModel()
        m.x = Var(initialize=3.0)
        m.y = Var(initialize=2.0)
        m.p = Param(initialize=1.0,mutable=True)
        m.e = Expression(initialize=1.0)

        # polynomial degree 0
        m.e.value = 0.0
        rep = generate_canonical_repn(m.e, {})
        rep1 = generate_canonical_repn(as_numeric(0.0), {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        rep = generate_canonical_repn(m.e+m.p, {})
        rep1 = generate_canonical_repn(0.0+m.p, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        m.e.value = 0.0+m.p
        rep = generate_canonical_repn(m.e+m.p**2, {})
        rep1 = generate_canonical_repn(0.0+m.p+m.p**2, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        # polynomial degree 1
        m.e.value = m.x
        rep = generate_canonical_repn(m.e, {})
        rep1 = generate_canonical_repn(m.x, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        rep = generate_canonical_repn(m.e+m.y, {})
        rep1 = generate_canonical_repn(m.x+m.y, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        m.e.value = 0.0
        rep = generate_canonical_repn(m.x+m.e, {})
        rep1 = generate_canonical_repn(m.x+0.0, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        m.e.value = 1.0
        rep = generate_canonical_repn(m.x*m.e, {})
        rep1 = generate_canonical_repn(m.x*1.0, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        # polynomial degree > 1
        m.e.value = m.x**2
        rep = generate_canonical_repn(m.e, {})
        rep1 = generate_canonical_repn(m.x**2, {})
        self.assertEqual(rep1, rep)

        rep = generate_canonical_repn(m.e+cos(m.y), {})
        rep1 = generate_canonical_repn(m.x**2+cos(m.y), {})
        self.assertEqual(rep1, rep)

        rep = generate_canonical_repn(m.e+cos(m.e), {})
        rep1 = generate_canonical_repn(m.x**2+cos(m.x**2), {})
        self.assertEqual(rep1, rep)

    def test_Expression_indexed(self):
        m = ConcreteModel()
        m.x = Var(initialize=3.0)
        m.y = Var(initialize=2.0)
        m.p = Param(initialize=1.0,mutable=True)
        m.e = Expression([1],initialize=1.0)

        # polynomial degree 0
        m.e[1].value = 0.0
        rep = generate_canonical_repn(m.e[1], {})
        rep1 = generate_canonical_repn(as_numeric(0.0), {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        rep = generate_canonical_repn(m.e[1]+m.p, {})
        rep1 = generate_canonical_repn(0.0+m.p, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        m.e[1].value = 0.0+m.p
        rep = generate_canonical_repn(m.e[1]+m.p**2, {})
        rep1 = generate_canonical_repn(0.0+m.p+m.p**2, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        # polynomial degree 1
        m.e[1].value = m.x
        rep = generate_canonical_repn(m.e[1], {})
        rep1 = generate_canonical_repn(m.x, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        rep = generate_canonical_repn(m.e[1]+m.y, {})
        rep1 = generate_canonical_repn(m.x+m.y, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        m.e[1].value = 0.0
        rep = generate_canonical_repn(m.x+m.e[1], {})
        rep1 = generate_canonical_repn(m.x+0.0, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        m.e[1].value = 1.0
        rep = generate_canonical_repn(m.x*m.e[1], {})
        rep1 = generate_canonical_repn(m.x*1.0, {})
        self.assertEqual(linear_repn_to_dict(rep1), linear_repn_to_dict(rep))

        # polynomial degree > 1
        m.e[1].value = m.x**2
        rep = generate_canonical_repn(m.e[1], {})
        rep1 = generate_canonical_repn(m.x**2, {})
        self.assertEqual(rep1, rep)

        rep = generate_canonical_repn(m.e[1]+cos(m.y), {})
        rep1 = generate_canonical_repn(m.x**2+cos(m.y), {})
        self.assertEqual(rep1, rep)

        rep = generate_canonical_repn(m.e[1]+cos(m.e[1]), {})
        rep1 = generate_canonical_repn(m.x**2+cos(m.x**2), {})
        self.assertEqual(rep1, rep)

    def test_Expr_if_constant(self):
        model = ConcreteModel()
        model.x = Var()
        model.x.fix(2.0)

        rep = generate_canonical_repn(Expr_if(IF=model.x, THEN=1, ELSE=-1))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear == None)
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.variables == None)
        baseline = {  None        : 1}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))
        rep = generate_canonical_repn(Expr_if(IF=model.x**2, THEN=1, ELSE=-1))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear == None)
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.variables == None)
        baseline = {  None        : 1}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))
        rep = generate_canonical_repn(Expr_if(IF=(1-cos(model.x-1)) > 0.5, THEN=1, ELSE=-1))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear == None)
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.variables == None)
        baseline = {  None        : -1}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))
        rep = generate_canonical_repn(Expr_if(IF=1, THEN=model.x, ELSE=-1))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear == None)
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.variables == None)
        baseline = {  None        : value(model.x)}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))
        rep = generate_canonical_repn(Expr_if(IF=0, THEN=1, ELSE=model.x))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear == None)
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.variables == None)
        baseline = {  None        : value(model.x)}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

    def test_Expr_if_linear(self):
        model = ConcreteModel()
        model.x = Var()
        model.y = Var()

        rep = generate_canonical_repn(Expr_if(IF=1, THEN=model.x+3*model.y+10, ELSE=-1))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant != None)
        self.assertTrue(rep.variables != None)
        baseline = { id(model.x) : 1,
                     id(model.y) : 3,
                     None        : 10}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))

        rep = generate_canonical_repn(Expr_if(IF=0.0, THEN=1.0, ELSE=-model.x))
        self.assertTrue(isinstance(rep, LinearCanonicalRepn) == True)
        self.assertTrue(rep.linear != None)
        self.assertTrue(rep.constant == None)
        self.assertTrue(rep.variables != None)
        baseline = { id(model.x) : -1}
        self.assertEqual(baseline,
                         linear_repn_to_dict(rep))


    def test_Expr_if_quadratic(self):
        model = ConcreteModel()
        model.x = Var()

        rep = generate_canonical_repn(Expr_if(IF=1.0, THEN=model.x**2, ELSE=-1.0))
        self.assertTrue(isinstance(rep, GeneralCanonicalRepn) == True)
        self.assertEqual(canonical_degree(rep), 2)
        rep = generate_canonical_repn(Expr_if(IF=0.0, THEN=1.0, ELSE=-model.x**2))
        self.assertTrue(isinstance(rep, GeneralCanonicalRepn) == True)
        self.assertEqual(canonical_degree(rep), 2)

    def test_Expr_if_nonlinear(self):
        model = ConcreteModel()
        model.x = Var()

        rep = generate_canonical_repn(Expr_if(IF=model.x, THEN=1.0, ELSE=-1.0))
        self.assertTrue(isinstance(rep, GeneralCanonicalRepn) == True)
        self.assertEqual(canonical_degree(rep), None)
        rep = generate_canonical_repn(Expr_if(IF=1.0,
                                              THEN=Expr_if(IF=model.x**2, THEN=1.0, ELSE=-1.0),
                                              ELSE=-1.0))
        self.assertTrue(isinstance(rep, GeneralCanonicalRepn) == True)
        self.assertEqual(canonical_degree(rep), None)
        rep = generate_canonical_repn(Expr_if(IF=model.x**2, THEN=1.0, ELSE=-1.0))
        self.assertTrue(isinstance(rep, GeneralCanonicalRepn) == True)
        self.assertEqual(canonical_degree(rep), None)

if __name__ == "__main__":
    unittest.main()
