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
# Test the standard expressions
#

import pickle
import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
import pyutilib.services

from pyomo.core.expr.current import Expr_if
from pyomo.core.expr import expr_common, current as EXPR
from pyomo.repn import *
from pyomo.environ import *
from pyomo.core.base.numvalue import native_numeric_types

from six import iteritems
from six.moves import range

class frozendict(dict):
    __slots__ = ('_hash',)
    def __hash__(self):
        rval = getattr(self, '_hash', None)
        if rval is None:
            rval = self._hash = hash(frozenset(iteritems(self)))
        return rval


# A utility to facilitate comparison of tuples where we don't care about ordering
def repn_to_dict(repn):
    result = {}
    for i in range(len(repn.linear_vars)):
        if id(repn.linear_vars[i]) in result:
            result[id(repn.linear_vars[i])] += repn.linear_coefs[i]
        else:
            result[id(repn.linear_vars[i])] = repn.linear_coefs[i]
    for i in range(len(repn.quadratic_vars)):
        v1_, v2_ = repn.quadratic_vars[i]
        if id(v1_) <= id(v2_):
            result[id(v1_), id(v2_)] = repn.quadratic_coefs[i]
        else:
            result[id(v2_), id(v1_)] = repn.quadratic_coefs[i]
    if not (repn.constant is None or (type(repn.constant) in native_numeric_types and repn.constant == 0)):
        result[None] = repn.constant
    return result


class TestSimple(unittest.TestCase):

    def test_number(self):
        # 1.0
        m = AbstractModel()
        m.a = Var()
        e = 1.0

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))
 
    def test_var(self):
        # a
        m = ConcreteModel()
        m.a = Var()
        e = m.a

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        m.a.value = 3
        m.a.fixed = True
        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None: 3 }
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None: 3 }
        self.assertEqual(baseline, repn_to_dict(rep))
        self.assertTrue(rep.constant is m.a)

    def test_param(self):
        # p
        m = AbstractModel()
        m.p = Param()
        e = m.p

        with self.assertRaises(ValueError):
            rep = generate_standard_repn(e)
        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None : m.p }
        self.assertEqual(baseline, repn_to_dict(rep))
        #s = pickle.dumps(rep)
        #rep = pickle.loads(s)
        #baseline = { None : m.p }
        #self.assertEqual(baseline, repn_to_dict(rep))

    def test_simplesum(self):
        # a + b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a + m.b
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]) : 1, id(rep.linear_vars[1]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_constsum(self):
        # a + 5
        m = AbstractModel()
        m.a = Var()
        e = m.a + 5
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # 5 + a
        m = AbstractModel()
        m.a = Var()
        e = 5 + m.a
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_paramsum(self):
        # a + 5
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(mutable=True, default=5)
        e = m.a + m.p
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # 5 + a
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(mutable=True, default=5)
        e = m.p + m.a
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        self.assertTrue(rep.constant is m.p)

    def test_paramprod1(self):
        # p*a
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(mutable=True, default=5)
        e = m.p*m.a
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]) : 5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p)

    def test_paramprod2(self):
        # p*a
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(mutable=True, default=0)
        e = m.p*m.a
 
        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { }
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 0 }
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p)

    def test_linear_sum1(self):
        #
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.p = Param(mutable=True, default=1)
        m.q = Param(mutable=True, default=2)
        e = m.p*m.x + m.q*m.y
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x):1, id(m.y):2 }
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x):1, id(m.y):2 }
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p)
        self.assertTrue(rep.linear_coefs[1] is m.q)

    def test_linear_sum2(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(5))
        m.x = Var(m.A)
        m.p = Param(m.A, mutable=True, default=1)
        e = quicksum(m.p[i]*m.x[i] for i in m.A)
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 5)
        self.assertTrue(len(rep.linear_coefs) == 5)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):1, id(m.x[2]):1, id(m.x[3]):1, id(m.x[4]):1}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 5)
        self.assertTrue(len(rep.linear_coefs) == 5)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):1, id(m.x[2]):1, id(m.x[3]):1, id(m.x[4]):1}
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p[0])
        self.assertTrue(rep.linear_coefs[1] is m.p[1])

    def test_linear_sum3(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(5))
        m.x = Var(m.A, initialize=3)
        m.p = Param(m.A, mutable=True, default=1)
        e = quicksum((i+1)*m.x[i] for i in m.A)
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 5)
        self.assertTrue(len(rep.linear_coefs) == 5)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):2, id(m.x[2]):3, id(m.x[3]):4, id(m.x[4]):5}
        self.assertEqual(baseline, repn_to_dict(rep))

        m.x[2].fixed = True

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):2, None:9, id(m.x[3]):4, id(m.x[4]):5}
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_linear_sum4(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(5))
        m.x = Var(m.A, initialize=3)
        m.p = Param(m.A, mutable=True, default=1)
        e = quicksum(m.p[i]*m.x[i] for i in m.A)
 
        m.x[2].fixed = True

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):1, None:3, id(m.x[3]):1, id(m.x[4]):1}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):1, None:3, id(m.x[3]):1, id(m.x[4]):1}
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p[0])
        self.assertTrue(rep.linear_coefs[1] is m.p[1])
        self.assertTrue(type(rep.constant) is EXPR.TermExpression)

    def test_linear_sum5(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(5))
        m.x = Var(m.A, initialize=3)
        m.p = Param(m.A, mutable=True, default=1)
        e = quicksum((m.p[i]*m.p[i])*m.x[i] for i in m.A)
 
        m.x[2].fixed = True

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):1, None:3, id(m.x[3]):1, id(m.x[4]):1}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):1, id(m.x[1]):1, None:3, id(m.x[3]):1, id(m.x[4]):1}
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0].is_expression_type())
        self.assertTrue(type(rep.constant) is EXPR.TermExpression)

    def test_linear_sum6(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(5))
        m.x = Var(m.A)
        m.p = Param(m.A, mutable=True, default=1)
        m.q = Param(m.A, mutable=True, default=2)
        e = quicksum(m.p[i]*m.x[i] if i < 5 else m.q[i-5]*m.x[i-5] for i in range(10))
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 5)
        self.assertTrue(len(rep.linear_coefs) == 5)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):3, id(m.x[1]):3, id(m.x[2]):3, id(m.x[3]):3, id(m.x[4]):3}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 5)
        self.assertTrue(len(rep.linear_coefs) == 5)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):3, id(m.x[1]):3, id(m.x[2]):3, id(m.x[3]):3, id(m.x[4]):3}
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0].is_expression_type())

    def test_general_sum1(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(3))
        m.x = Var(m.A, initialize=2)
        m.p = Param(m.A, mutable=True, default=3)
        e = sum(m.p[i]*m.x[i] for i in range(3))
        m.x[1].fixed = True
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):3, None:6, id(m.x[2]):3}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):3, None:6, id(m.x[2]):3}
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p[0])

    def test_general_sum2(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(3))
        m.x = Var(m.A, initialize=2)
        m.p = Param(m.A, mutable=True, default=3)
        e = sum(m.p[i]*m.x[i] if i!=1 else m.x[i] for i in range(3))
        m.x[1].fixed = True
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):3, None:2, id(m.x[2]):3}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):3, None:2, id(m.x[2]):3}
        self.assertEqual(baseline, repn_to_dict(rep))
        #
        self.assertTrue(rep.linear_coefs[0] is m.p[0])

    def test_general_sum3(self):
        #
        m = ConcreteModel()
        m.A = Set(initialize=range(3))
        m.x = Var(m.A, initialize=2)
        m.p = Param(m.A, mutable=True, default=3)
        e = sum(m.p[i]*m.x[i] if i<3 else m.x[i-3] for i in range(6))
 
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):4, id(m.x[1]):4, id(m.x[2]):4}
        self.assertEqual(baseline, repn_to_dict(rep))

        rep = generate_standard_repn(e, compute_values=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.x[0]):4, id(m.x[1]):4, id(m.x[2]):4}
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_nestedSum(self):
        #
        # Check the structure of nested sums
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #           +
        #          / \
        #         +   5
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = e1 + 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1, id(m.b) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       + 
        #      / \ 
        #     5   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = 5 + e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : 1, id(m.b) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #           +
        #          / \
        #         +   c
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = e1 + m.c

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):1, id(rep.linear_vars[2]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       + 
        #      / \ 
        #     c   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = m.c + e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):1, id(rep.linear_vars[2]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            +
        #          /   \
        #         +     +
        #        / \   / \
        #       a   b c   d
        e1 = m.a + m.b
        e2 = m.c + m.d
        e = e1 + e2

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : 1, id(m.c) : 1, id(m.d) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):1, id(rep.linear_vars[2]):1, id(rep.linear_vars[3]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_sumOf_nestedTrivialProduct(self):
        #
        # Check sums with nested products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()

        #       +
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 + m.b

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]) : 5, id(rep.linear_vars[1]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       +
        #      / \
        #     b   *
        #        / \
        #       a   5
        e = m.b + e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[1]) : 5, id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            +
        #          /   \
        #         *     +
        #        / \   / \
        #       a   5 b   c
        e2 = m.b + m.c
        e = e1 + e2

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[2]) : 5, id(rep.linear_vars[0]) : 1, id(rep.linear_vars[1]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            +
        #          /   \
        #         +     *
        #        / \   / \
        #       b   c a   5
        e2 = m.b + m.c
        e = e2 + e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 1, id(m.c) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[2]) : 5, id(rep.linear_vars[0]) : 1, id(rep.linear_vars[1]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            +
        #          /   \
        #         *     *
        #        / \   / \
        #       a   5 b   5
        e2 = m.b * 5
        e = e2 + e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 5, id(m.b) : 5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_negation(self):
        #    -
        #     \
        #      a
        m = AbstractModel()
        m.a = Var()
        e = - m.a

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : -1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]) : -1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_simpleDiff(self):
        #    -
        #   / \
        #  a   b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        e = m.a - m.b

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a) : 1, id(m.b) : -1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]) : 1, id(rep.linear_vars[1]) : -1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #    -
        #   / \
        #  a   a
        e = m.a - m.a

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertEqual(len(rep.linear_vars), 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_constDiff(self):
        #    -
        #   / \
        #  a   5
        m = AbstractModel()
        m.a = Var()
        e = m.a - 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:-5, id(m.a) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:-5, id(rep.linear_vars[0]) : 1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #    -
        #   / \
        #  5   a
        e = 5 - m.a

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a) : -1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_nestedDiff(self):
        #
        # Check the structure of nested differences
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        #       -
        #      / \
        #     -   5
        #    / \
        #   a   b
        e1 = m.a - m.b
        e = e1 - 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:-5, id(m.a):1, id(m.b):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:-5, id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       -
        #      / \
        #     5   -
        #        / \
        #       a   b
        e1 = m.a - m.b
        e = 5 - e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:5, id(m.a):-1, id(m.b):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:5, id(rep.linear_vars[0]):-1, id(rep.linear_vars[1]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       -
        #      / \
        #     -   c
        #    / \
        #   a   b
        e1 = m.a - m.b
        e = e1 - m.c

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):1, id(m.b):-1, id(m.c):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):-1, id(rep.linear_vars[2]):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       -
        #      / \
        #     c   -
        #        / \
        #       a   b
        e1 = m.a - m.b
        e = m.c - e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):-1, id(m.b):1, id(m.c):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[1]):-1, id(rep.linear_vars[0]):1, id(rep.linear_vars[2]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            -
        #          /   \
        #         -     -
        #        / \   / \
        #       a   b c   d
        e1 = m.a - m.b
        e2 = m.c - m.d
        e = e1 - e2

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):1, id(m.b):-1, id(m.c):-1, id(m.d):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):-1, id(rep.linear_vars[2]):-1, id(rep.linear_vars[3]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            -
        #          /   \
        #         -     -
        #        / \   / \
        #       c   d a   b
        e1 = m.a - m.b
        e2 = m.c - m.d
        e = e2 - e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 4)
        self.assertTrue(len(rep.linear_coefs) == 4)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):-1, id(m.b):1, id(m.c):1, id(m.d):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[2]):-1, id(rep.linear_vars[3]):1, id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_sumOf_nestedTrivialProduct2(self):
        #
        # Check the structure of sum of products
        #
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()

        #       -
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 - m.b

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       -
        #      / \
        #     b   *
        #        / \
        #       a   5
        e1 = m.a * 5
        e = m.b - e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):-5, id(m.b):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[1]):-5, id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            -
        #          /   \
        #         *     -
        #        / \   / \
        #       a   5 b   c
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e1 - e2

        rep = generate_standard_repn(e)
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):-1, id(m.c):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):-1, id(rep.linear_vars[2]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            -
        #          /   \
        #         -     *
        #        / \   / \
        #       b   c a   5
        e1 = m.a * 5
        e2 = m.b - m.c
        e = e2 - e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 3)
        self.assertTrue(len(rep.linear_coefs) == 3)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):-5, id(m.b):1, id(m.c):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[2]):-5, id(rep.linear_vars[0]):1, id(rep.linear_vars[1]):-1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       -
        #        \
        #         -
        #        / \
        #       a   b
        e = - (m.a - m.b)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):-1, id(m.b):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):-1, id(rep.linear_vars[1]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_simpleProduct1(self):
        m = ConcreteModel()
        m.a = Var()
        m.p = Param(default=2)
        #    *
        #   / \
        #  a   p
        e = m.a * m.p

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):2 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):2 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #    *
        #   / \
        #  a   0
        e = m.a * 0

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_simpleProduct2(self):
        #    *
        #   / \
        #  a   5
        m = AbstractModel()
        m.a = Var()
        e = m.a * 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #    *
        #   / \
        #  5   a
        e = 5 * m.a

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_nestedProduct(self):
        #       *
        #      / \
        #     *   5
        #    / \
        #   a   b
        m = ConcreteModel()
        m.a = Var()
        m.b = Param(default=2)
        m.c = Param(default=3)
        m.d = Param(default=7)

        e1 = m.a * m.b
        e = e1 * 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):10 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):10 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #      / \
        #     5   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = 5 * e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):10 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):10 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       a   b c   d
        e1 = m.a * m.b
        e2 = m.c * m.d
        e = e1 * e2

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):42 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):42 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_nestedProduct2(self):
        #
        # Check the structure of nested products
        #
        m = ConcreteModel()
        m.a = Param(default=2)
        m.b = Param(default=3)
        m.c = Param(default=5)
        m.d = Var()
        #
        # Check the structure of nested products
        #
        #            *
        #          /   \
        #         +     +
        #        / \   / \
        #       c    +    d
        #           / \
        #          a   b
        e1 = m.a + m.b
        e2 = m.c + e1
        e3 = e1 + m.d
        e = e2 * e3

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:50, id(m.d):10 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { None:50, id(rep.linear_vars[0]):10 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #
        # Check the structure of nested products
        #
        #            *
        #          /   \
        #         *     *
        #        / \   / \
        #       c    +    d
        #           / \
        #          a   b
        e1 = m.a + m.b
        e2 = m.c * e1
        e3 = e1 * m.d
        e = e2 * e3

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.d):125 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):125 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_division(self):
        #
        #           /
        #          / \
        #         +   2
        #        / \
        #       a   b
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        m.y = Var(initialize=2.0)
        m.y.fixed = True

        e = (m.a + m.b)/2.0

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):0.5, id(m.b):0.5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):0.5, id(rep.linear_vars[1]):0.5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #           /
        #          / \
        #         +   y
        #        / \
        #       a   b
        e = (m.a + m.b)/m.y

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):0.5, id(m.b):0.5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):0.5, id(rep.linear_vars[1]):0.5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #            /
        #          /   \
        #         +     +
        #        / \   / \
        #       a   b y   2
        e = (m.a + m.b)/(m.y+2)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):0.25, id(m.b):0.25 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):0.25, id(rep.linear_vars[1]):0.25 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_weighted_sum1(self):
        #       *
        #      / \
        #     +   5
        #    / \
        #   a   b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        e1 = m.a + m.b
        e = e1 * 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #      / \
        #     5   +
        #        / \
        #       a   b
        e1 = m.a + m.b
        e = 5 * e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):5, id(m.b):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #      / \
        #     5   *
        #        / \
        #       2   +
        #          / \
        #         a   b
        e1 = m.a + m.b
        e = 5 * 2* e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):10, id(m.b):10 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):10, id(rep.linear_vars[1]):10 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       5(a+2(a+b))
        e = 5*(m.a+2*(m.a+m.b))

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):15, id(m.b):10 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):15, id(rep.linear_vars[1]):10 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_quadratic1(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        ab_key = (id(m.a),id(m.b)) if id(m.a) <= id(m.b) else (id(m.b),id(m.a))

        #       *
        #      / \
        #     *   b
        #    / \
        #   a   5
        e1 = m.a * 5
        e = e1 * m.b

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
            baseline = { (id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])):5 }
        else:
            baseline = { (id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #      / \
        #     *   5
        #    / \
        #   a   b
        e1 = m.a * m.b
        e = e1 * 5

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
            baseline = { (id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])):5 }
        else:
            baseline = { (id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #      / \
        #     5   *
        #        / \
        #       a   b
        e1 = m.a * m.b
        e = 5*e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
            baseline = { (id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])):5 }
        else:
            baseline = { (id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #      / \
        #     b   *
        #        / \
        #       a   5
        e1 = m.a * 5
        e = m.b*e1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
            baseline = { (id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])):5 }
        else:
            baseline = { (id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])):5 }
        self.assertEqual(baseline, repn_to_dict(rep))


    def test_quadratic2(self):
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()

        ab_key = (id(m.a),id(m.b)) if id(m.a) <= id(m.b) else (id(m.b),id(m.a))
        ac_key = (id(m.a),id(m.c)) if id(m.a) <= id(m.c) else (id(m.c),id(m.a))
        bc_key = (id(m.b),id(m.c)) if id(m.b) <= id(m.c) else (id(m.c),id(m.b))

        #       *
        #      / \
        #     +   b
        #    / \
        #   a   5
        e1 = m.a + 5
        e = e1 * m.b

        # Collect quadratics
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:1, id(m.b):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
            baseline = { (id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])):1, id(rep.linear_vars[0]):5 }
        else:
            baseline = { (id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])):1, id(rep.linear_vars[0]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # Do not collect quadratics
        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 2)
        baseline1 = { }
        self.assertEqual(baseline1, repn_to_dict(rep))
        baseline2 = set([ id(m.a), id(m.b) ])
        self.assertEqual(baseline2, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline1, repn_to_dict(rep))

        #       *
        #      / \
        #     b   +
        #        / \
        #       a   5
        e1 = m.a + 5
        e = m.b * e1

        # Collect quadratics
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:1, id(m.b):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        if id(rep.quadratic_vars[0][0]) < id(rep.quadratic_vars[0][1]):
            baseline = { (id(rep.quadratic_vars[0][0]), id(rep.quadratic_vars[0][1])):1, id(rep.linear_vars[0]):5 }
        else:
            baseline = { (id(rep.quadratic_vars[0][1]), id(rep.quadratic_vars[0][0])):1, id(rep.linear_vars[0]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # Do not collect quadratics
        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 2)
        baseline1 = { }
        self.assertEqual(baseline1, repn_to_dict(rep))
        baseline2 = set([ id(m.a), id(m.b) ])
        self.assertEqual(baseline2, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline1, repn_to_dict(rep))

        #       *
        #     /   \
        #    +     +
        #   / \   / \
        #  c   b a   5
        e = (m.c + m.b) * (m.a + 5)

        # Collect quadratics
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 2)
        self.assertTrue(len(rep.quadratic_coefs) == 2)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:1, ac_key:1, id(m.b):5, id(m.c):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        ab_key_ = (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])) if id(rep.quadratic_vars[0][0]) <= id(rep.quadratic_vars[0][1]) else (id(rep.quadratic_vars[0][1]),id(rep.quadratic_vars[0][0]))
        ac_key_ = (id(rep.quadratic_vars[1][0]),id(rep.quadratic_vars[1][1])) if id(rep.quadratic_vars[1][0]) <= id(rep.quadratic_vars[1][1]) else (id(rep.quadratic_vars[1][1]),id(rep.quadratic_vars[1][0]))
        baseline = { ab_key_:1, ac_key_:1, id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       *
        #     /   \
        #    +     +
        #   / \   / \
        #  a   5 b   c
        e = (m.a + 5) * (m.b + m.c)

        # Collect quadratics
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 2)
        self.assertTrue(len(rep.linear_coefs) == 2)
        self.assertTrue(len(rep.quadratic_vars) == 2)
        self.assertTrue(len(rep.quadratic_coefs) == 2)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:1, ac_key:1, id(m.b):5, id(m.c):5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        ab_key_ = (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])) if id(rep.quadratic_vars[0][0]) <= id(rep.quadratic_vars[0][1]) else (id(rep.quadratic_vars[0][1]),id(rep.quadratic_vars[0][0]))
        ac_key_ = (id(rep.quadratic_vars[1][0]),id(rep.quadratic_vars[1][1])) if id(rep.quadratic_vars[1][0]) <= id(rep.quadratic_vars[1][1]) else (id(rep.quadratic_vars[1][1]),id(rep.quadratic_vars[1][0]))
        baseline = { ab_key_:1, ac_key_:1, id(rep.linear_vars[0]):5, id(rep.linear_vars[1]):5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # Do not collect quadratics
        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 3)
        baseline1 = { }
        self.assertEqual(baseline1, repn_to_dict(rep))
        baseline2 = set([ id(m.a), id(m.b), id(m.c) ])
        self.assertEqual(baseline2, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline1, repn_to_dict(rep))

        #       *
        #     /   \
        #    *     +
        #   / \   / \
        #  a   5 b   c
        e = (m.a * 5) * (m.b + m.c)

        # Collect quadratics
        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 2)
        self.assertTrue(len(rep.quadratic_coefs) == 2)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:5, ac_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        ab_key_ = (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])) if id(rep.quadratic_vars[0][0]) <= id(rep.quadratic_vars[0][1]) else (id(rep.quadratic_vars[0][1]),id(rep.quadratic_vars[0][0]))
        ac_key_ = (id(rep.quadratic_vars[1][0]),id(rep.quadratic_vars[1][1])) if id(rep.quadratic_vars[1][0]) <= id(rep.quadratic_vars[1][1]) else (id(rep.quadratic_vars[1][1]),id(rep.quadratic_vars[1][0]))
        baseline = { ab_key_:5, ac_key_:5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # Do not collect quadratics
        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 3)
        baseline1 = { }
        self.assertEqual(baseline1, repn_to_dict(rep))
        baseline2 = set([ id(m.a), id(m.b), id(m.c) ])
        self.assertEqual(baseline2, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline1, repn_to_dict(rep))

        #       *
        #     /   \
        #    +     *
        #   / \   / \
        #  b   c a   5
        e = (m.b + m.c) * (m.a * 5)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 2)
        self.assertTrue(len(rep.quadratic_coefs) == 2)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { ab_key:5, ac_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        ab_key_ = (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])) if id(rep.quadratic_vars[0][0]) <= id(rep.quadratic_vars[0][1]) else (id(rep.quadratic_vars[0][1]),id(rep.quadratic_vars[0][0]))
        ac_key_ = (id(rep.quadratic_vars[1][0]),id(rep.quadratic_vars[1][1])) if id(rep.quadratic_vars[1][0]) <= id(rep.quadratic_vars[1][1]) else (id(rep.quadratic_vars[1][1]),id(rep.quadratic_vars[1][0]))
        baseline = { ab_key_:5, ac_key_:5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # Do not collect quadratics
        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 3)
        baseline1 = { }
        self.assertEqual(baseline1, repn_to_dict(rep))
        baseline2 = set([ id(m.a), id(m.b), id(m.c) ])
        self.assertEqual(baseline2, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline1, repn_to_dict(rep))

        #       *
        #     /   \
        #    +     *
        #   / \   / \
        #  a   5 b   c
        e = (m.a + 5) * (m.b * m.c)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 3)
        baseline = { bc_key:5 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        bc_key_ = (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])) if id(rep.quadratic_vars[0][0]) <= id(rep.quadratic_vars[0][1]) else (id(rep.quadratic_vars[0][1]),id(rep.quadratic_vars[0][0]))
        baseline = { bc_key_:5 }
        self.assertEqual(baseline, repn_to_dict(rep))

        # Do not collect quadratics
        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertEqual(len(rep.nonlinear_vars), 3)
        baseline1 = { }
        self.assertEqual(baseline1, repn_to_dict(rep))
        baseline2 = set([ id(m.a), id(m.b), id(m.c) ])
        self.assertEqual(baseline2, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline1, repn_to_dict(rep))

    def test_pow(self):
        #       ^
        #      / \
        #     a   0
        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        m.p = Param()
        m.q = Param(default=1)
        m.r = Param(default=2)

        e = m.a ** 0

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ^
        #      / \
        #     a   1
        e = m.a ** 1

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ^
        #      / \
        #     a   2
        e = m.a ** 2

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { (id(m.a),id(m.a)):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ^
        #      / \
        #     a   r
        e = m.a ** m.r

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 2 )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertTrue( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 1)
        self.assertTrue(len(rep.quadratic_coefs) == 1)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { (id(m.a),id(m.a)):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { (id(rep.quadratic_vars[0][0]),id(rep.quadratic_vars[0][1])):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ^
        #      / \
        #     a   2
        e = m.a ** 2

        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = set([ id(rep.nonlinear_vars[0]) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))

        #       ^
        #      / \
        #     a   m.r 
        e = m.a ** m.r

        rep = generate_standard_repn(e, quadratic=False)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = set([ id(rep.nonlinear_vars[0]) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))

        #       ^
        #      / \
        #     a   q
        e = m.a ** m.q

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_abs(self):
        #      abs
        #      / 
        #     a   
        m = ConcreteModel()
        m.a = Var()
        m.q = Param(default=-1)

        e = abs(m.a)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = set([ id(rep.nonlinear_vars[0]) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))

        #      abs
        #      / 
        #     a   
        e = abs(m.a)
        m.a.set_value(-1)
        m.a.fixed = True

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

        #      abs
        #      / 
        #     q   
        e = abs(m.q)

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_cos(self):
        #      cos
        #      / 
        #     a   
        m = ConcreteModel()
        m.a = Var()
        m.q = Param(default=0)

        e = cos(m.a)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 1)
        baseline = set([ id(m.a) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = set([ id(rep.nonlinear_vars[0]) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))

        #      cos
        #      / 
        #     a   
        e = cos(m.a)
        m.a.set_value(0)
        m.a.fixed = True

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:1.0 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

        #      cos
        #      / 
        #     q   
        e = cos(m.q)

        rep = generate_standard_repn(e)
        #
        self.assertTrue( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 0 )
        self.assertTrue( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { None:1.0 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        self.assertEqual(baseline, repn_to_dict(rep))

    def test_ExprIf(self):
        #       ExprIf
        #      /  |   \
        #   True  a    b
        m = AbstractModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.q = Param(default=1)

        e = EXPR.Expr_if(IF_=True, THEN_=m.a, ELSE_=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ExprIf
        #      /  |   \
        #  False  a    b
        e = EXPR.Expr_if(IF_=False, THEN_=m.a, ELSE_=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.b):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ExprIf
        #      /  |   \
        #     c   a    b
        e = EXPR.Expr_if(IF_=m.c, THEN_=m.a, ELSE_=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), None )
        self.assertFalse( rep.is_constant() )
        self.assertFalse( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertTrue( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 0)
        self.assertTrue(len(rep.linear_coefs) == 0)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertFalse(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 3)
        baseline = set([ id(m.a), id(m.b), id(m.c) ])
        self.assertEqual(baseline, set(id(v_) for v_ in EXPR.identify_variables(rep.nonlinear_expr)))
        #s = pickle.dumps(rep)
        #rep = pickle.loads(s)
        #self.assertEqual(baseline, repn_to_dict(rep))

        m = ConcreteModel()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.q = Param(default=1)

        #       ExprIf
        #      /  |   \
        #  bool  a    b
        e = EXPR.Expr_if(IF_=m.q, THEN_=m.a, ELSE_=m.b)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.a):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))

        #       ExprIf
        #      /  |   \
        #     c   a    b
        e = EXPR.Expr_if(IF_=m.c, THEN_=m.a, ELSE_=m.b)
        m.c.fixed = True
        m.c.set_value(0)

        rep = generate_standard_repn(e)
        #
        self.assertFalse( rep.is_fixed() )
        self.assertEqual( rep.polynomial_degree(), 1 )
        self.assertFalse( rep.is_constant() )
        self.assertTrue( rep.is_linear() )
        self.assertFalse( rep.is_quadratic() )
        self.assertFalse( rep.is_nonlinear() )
        #
        self.assertTrue(len(rep.linear_vars) == 1)
        self.assertTrue(len(rep.linear_coefs) == 1)
        self.assertTrue(len(rep.quadratic_vars) == 0)
        self.assertTrue(len(rep.quadratic_coefs) == 0)
        self.assertTrue(rep.nonlinear_expr is None)
        self.assertTrue(len(rep.nonlinear_vars) == 0)
        baseline = { id(m.b):1 }
        self.assertEqual(baseline, repn_to_dict(rep))
        s = pickle.dumps(rep)
        rep = pickle.loads(s)
        baseline = { id(rep.linear_vars[0]):1 }
        self.assertEqual(baseline, repn_to_dict(rep))


if __name__ == "__main__":
    unittest.main()
