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
# Unit Tests for Port
#

import pyutilib.th as unittest
from six import StringIO

from pyomo.environ import ConcreteModel, AbstractModel, Var, Set, NonNegativeReals, Binary, Reals, Integers, RangeSet
from pyomo.network import Port, Arc

class TestPort(unittest.TestCase):

    def test_default_scalar_constructor(self):
        model = ConcreteModel()
        model.c = Port()
        self.assertEqual(len(model.c), 1)
        self.assertEqual(len(model.c.vars), 0)

        model = AbstractModel()
        model.c = Port()
        self.assertEqual(len(model.c), 0)
        # FIXME: Not sure I like this behavior: but since this is
        # (currently) an attribute, there is no way to check for
        # construction withough converting it to a property.
        #
        # TODO: if we move away from multiple inheritance for
        # simplevars, then this can trigger an exception (cleanly)
        self.assertEqual(len(model.c.vars), 0)

        inst = model.create_instance()
        self.assertEqual(len(inst.c), 1)
        self.assertEqual(len(inst.c.vars), 0)

    def test_default_indexed_constructor(self):
        model = ConcreteModel()
        model.c = Port([1, 2, 3])
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.c[1].vars), 0)

        model = AbstractModel()
        model.c = Port([1, 2, 3])
        self.assertEqual(len(model.c), 0)
        self.assertRaises(ValueError, model.c.__getitem__, 1)

        inst = model.create_instance()
        self.assertEqual(len(inst.c), 3)
        self.assertEqual(len(inst.c[1].vars), 0)

    def test_add_scalar_vars(self):
        pipe = ConcreteModel()
        pipe.flow = Var()
        pipe.pIn  = Var( within=NonNegativeReals )
        pipe.pOut  = Var( within=NonNegativeReals )
  
        pipe.OUT = Port()
        pipe.OUT.add(pipe.flow, "flow")
        pipe.OUT.add(pipe.pOut, "pressure")
        self.assertEqual(len(pipe.OUT), 1)
        self.assertEqual(len(pipe.OUT.vars), 2)
        self.assertFalse(pipe.OUT.vars['flow'].is_expression_type())

        pipe.IN = Port()
        pipe.IN.add(-pipe.flow, "flow")
        pipe.IN.add(pipe.pIn, "pressure")
        self.assertEqual(len(pipe.IN), 1)
        self.assertEqual(len(pipe.IN.vars), 2)
        self.assertTrue(pipe.IN.vars['flow'].is_expression_type())
        
    def test_add_indexed_vars(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Port()
        pipe.OUT.add(pipe.flow, "flow")
        pipe.OUT.add(pipe.composition, "composition")
        pipe.OUT.add(pipe.pIn, "pressure")

        self.assertEqual(len(pipe.OUT), 1)
        self.assertEqual(len(pipe.OUT.vars), 3)

    def test_fixed(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Port()
        self.assertTrue( pipe.OUT.is_fixed())

        pipe.OUT.add(pipe.flow, "flow")
        self.assertFalse( pipe.OUT.is_fixed())

        pipe.flow.fix(0)
        self.assertTrue( pipe.OUT.is_fixed())

        pipe.OUT.add(-pipe.pIn, "pressure")
        self.assertFalse( pipe.OUT.is_fixed())

        pipe.pIn.fix(1)
        self.assertTrue( pipe.OUT.is_fixed())

        pipe.OUT.add(pipe.composition, "composition")
        self.assertFalse( pipe.OUT.is_fixed())

        pipe.composition['a'].fix(1)
        self.assertFalse( pipe.OUT.is_fixed())

        pipe.composition['b'].fix(1)
        pipe.composition['c'].fix(1)
        self.assertTrue( pipe.OUT.is_fixed())

        m = ConcreteModel()
        m.SPECIES = Set(initialize=['a','b','c'])
        m.flow = Var()
        m.composition = Var(m.SPECIES)
        m.pIn  = Var( within=NonNegativeReals )

        m.port = Port()
        m.port.add(m.flow, "flow")
        m.port.add(-m.pIn, "pressure")
        m.port.add(m.composition, "composition")
        m.port.fix()
        self.assertTrue(m.port.is_fixed())

    def test_polynomial_degree(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Port()
        self.assertEqual( pipe.OUT.polynomial_degree(), 0)

        pipe.OUT.add(pipe.flow, "flow")
        self.assertEqual( pipe.OUT.polynomial_degree(), 1)

        pipe.flow.fix(0)
        self.assertEqual( pipe.OUT.polynomial_degree(), 0)

        pipe.OUT.add(-pipe.pIn, "pressure")
        self.assertEqual( pipe.OUT.polynomial_degree(), 1)

        pipe.pIn.fix(1)
        self.assertEqual( pipe.OUT.polynomial_degree(), 0)

        pipe.OUT.add(pipe.composition, "composition")
        self.assertEqual( pipe.OUT.polynomial_degree(), 1)

        pipe.composition['a'].fix(1)
        self.assertEqual( pipe.OUT.polynomial_degree(), 1)

        pipe.composition['b'].fix(1)
        pipe.composition['c'].fix(1)
        self.assertEqual( pipe.OUT.polynomial_degree(), 0)

        pipe.OUT.add(pipe.flow*pipe.pIn, "quadratic")
        self.assertEqual( pipe.OUT.polynomial_degree(), 0)

        pipe.flow.unfix()
        self.assertEqual( pipe.OUT.polynomial_degree(), 1)

        pipe.pIn.unfix()
        self.assertEqual( pipe.OUT.polynomial_degree(), 2)

        pipe.OUT.add(pipe.flow/pipe.pIn, "nonLin")
        self.assertEqual( pipe.OUT.polynomial_degree(), None)

    def test_potentially_variable(self):
        m = ConcreteModel()
        m.x = Var()
        m.p = Port()
        self.assertTrue(m.p.is_potentially_variable())
        m.p.add(-m.x)
        self.assertTrue(m.p.is_potentially_variable())

    def test_binary(self):
        m = ConcreteModel()
        m.x = Var(domain=Binary)
        m.y = Var(domain=Reals)
        m.p = Port()

        self.assertTrue(m.p.is_binary())

        m.p.add(m.x)
        self.assertTrue(m.p.is_binary())

        m.p.add(-m.x, "foo")
        self.assertTrue(m.p.is_binary())

        m.p.add(m.y)
        self.assertFalse(m.p.is_binary())

        m.p.remove('y')
        self.assertTrue(m.p.is_binary())

        m.p.add(-m.y, "bar")
        self.assertFalse(m.p.is_binary())

    def test_integer(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers)
        m.y = Var(domain=Reals)
        m.p = Port()

        self.assertTrue(m.p.is_integer())

        m.p.add(m.x)
        self.assertTrue(m.p.is_integer())

        m.p.add(-m.x, "foo")
        self.assertTrue(m.p.is_integer())

        m.p.add(m.y)
        self.assertFalse(m.p.is_integer())

        m.p.remove('y')
        self.assertTrue(m.p.is_integer())

        m.p.add(-m.y, "bar")
        self.assertFalse(m.p.is_integer())

    def test_continuous(self):
        m = ConcreteModel()
        m.x = Var(domain=Reals)
        m.y = Var(domain=Integers)
        m.p = Port()

        self.assertTrue(m.p.is_continuous())

        m.p.add(m.x)
        self.assertTrue(m.p.is_continuous())

        m.p.add(-m.x, "foo")
        self.assertTrue(m.p.is_continuous())

        m.p.add(m.y)
        self.assertFalse(m.p.is_continuous())

        m.p.remove('y')
        self.assertTrue(m.p.is_continuous())

        m.p.add(-m.y, "bar")
        self.assertFalse(m.p.is_continuous())

    def test_getattr(self):
        m = ConcreteModel()
        m.x = Var()
        m.port = Port()
        m.port.add(m.x)
        self.assertIs(m.port.x, m.x)

    def test_arc_lists(self):
        m = ConcreteModel()
        m.x = Var()
        m.p1 = Port()
        m.p2 = Port()
        m.p3 = Port()
        m.p4 = Port()
        m.p5 = Port()
        m.p1.add(m.x)
        m.p2.add(m.x)
        m.p3.add(m.x)
        m.p4.add(m.x)
        m.p5.add(m.x)
        m.a1 = Arc(source=m.p1, destination=m.p2)
        m.a2 = Arc(source=m.p1, destination=m.p3)
        m.a3 = Arc(source=m.p4, destination=m.p1)
        m.a4 = Arc(source=m.p5, destination=m.p1)

        self.assertEqual(len(m.p1.dests()), 2)
        self.assertEqual(len(m.p1.sources()), 2)
        self.assertEqual(len(m.p1.arcs()), 4)
        self.assertEqual(len(m.p2.dests()), 0)
        self.assertEqual(len(m.p2.sources()), 1)
        self.assertEqual(len(m.p2.arcs()), 1)
        self.assertIn(m.a1, m.p1.dests())
        self.assertIn(m.a1, m.p2.sources())
        self.assertNotIn(m.a1, m.p1.sources())
        self.assertNotIn(m.a1, m.p2.dests())

        self.assertEqual(len(m.p1.dests(active=True)), 2)
        self.assertEqual(len(m.p1.sources(active=True)), 2)
        self.assertEqual(len(m.p1.arcs(active=True)), 4)
        self.assertEqual(len(m.p2.dests(active=True)), 0)
        self.assertEqual(len(m.p2.sources(active=True)), 1)
        self.assertEqual(len(m.p2.arcs(active=True)), 1)
        self.assertIn(m.a1, m.p1.dests(active=True))
        self.assertIn(m.a1, m.p2.sources(active=True))
        self.assertNotIn(m.a1, m.p1.sources(active=True))
        self.assertNotIn(m.a1, m.p2.dests(active=True))

        m.a2.deactivate()

        self.assertNotIn(m.a2, m.p1.dests(active=True))
        self.assertNotIn(m.a2, m.p3.sources(active=True))
        self.assertIn(m.a2, m.p1.dests(active=False))
        self.assertIn(m.a2, m.p3.sources(active=False))
        self.assertIn(m.a2, m.p1.arcs(active=False))
        self.assertIn(m.a2, m.p3.arcs(active=False))
        self.assertIn(m.a2, m.p1.dests())
        self.assertIn(m.a2, m.p3.sources())
        self.assertIn(m.a2, m.p1.arcs())
        self.assertIn(m.a2, m.p3.arcs())

    def test_remove(self):
        m = ConcreteModel()
        m.x = Var()
        m.port = Port()
        m.port.add(m.x)
        self.assertIn('x', m.port.vars)
        self.assertIn('x', m.port._rules)
        m.port.remove('x')
        self.assertNotIn('x', m.port.vars)
        self.assertNotIn('x', m.port._rules)

    def test_extends(self):
        m = ConcreteModel()
        m.x = Var()
        m.p1 = Port()
        m.p1.add(m.x, rule=Port.Extensive)
        m.p2 = Port(extends=m.p1)
        self.assertIs(m.p2.x, m.x)
        self.assertIs(m.p2.rule_for('x'), Port.Extensive)
        self.assertTrue(m.p2.is_extensive('x'))
        self.assertFalse(m.p2.is_equality('x'))

    def test_add_from_containers(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.p1 = Port(initialize=[m.x, m.y])
        m.p2 = Port(initialize=[(m.x, Port.Equality), (m.y, Port.Extensive)])
        m.p3 = Port(initialize=dict(this=m.x, that=m.y))
        m.p4 = Port(initialize=dict(this=(m.x, Port.Equality),
                                    that=(m.y, Port.Extensive)))

        self.assertIs(m.p1.x, m.x)
        self.assertIs(m.p1.y, m.y)
        self.assertIs(m.p2.x, m.x)
        self.assertTrue(m.p2.is_equality('x'))
        self.assertIs(m.p2.y, m.y)
        self.assertTrue(m.p2.is_extensive('y'))
        self.assertIs(m.p3.this, m.x)
        self.assertIs(m.p3.that, m.y)
        self.assertIs(m.p4.this, m.x)
        self.assertTrue(m.p4.is_equality('this'))
        self.assertIs(m.p4.that, m.y)
        self.assertTrue(m.p4.is_extensive('that'))

    def test_fix_unfix(self):
        m = ConcreteModel()
        m.x = Var()
        m.port = Port()
        m.port.add(m.x)
        m.x.value = 10
        m.port.fix()
        self.assertTrue(m.x.is_fixed())
        m.port.unfix()
        self.assertFalse(m.x.is_fixed())

    def test_iter_vars(self):
        def contains(item, container):
            # use this instead of "in" to avoid "==" operation
            return any(item is mem for mem in container)

        m = ConcreteModel()
        m.s = RangeSet(5)
        m.x = Var()
        m.y = Var()
        m.z = Var(m.s)
        m.a = Var()
        m.b = Var()
        m.c = Var()
        expr = m.a + m.b * m.c
        p = m.p = Port()

        v = list(p.iter_vars())
        self.assertEqual(len(v), 0)

        p.add(m.x)
        v = list(p.iter_vars())
        self.assertEqual(len(v), 1)

        p.add(m.y)
        v = list(p.iter_vars())
        self.assertEqual(len(v), 2)

        p.add(m.z)
        v = list(p.iter_vars())
        self.assertEqual(len(v), 7)
        self.assertTrue(contains(m.x, v))
        self.assertTrue(contains(m.z[3], v))

        p.add(expr, "expr")
        v = list(p.iter_vars())
        self.assertEqual(len(v), 8)
        self.assertTrue(contains(m.x, v))
        self.assertTrue(contains(m.z[3], v))
        self.assertTrue(contains(expr, v))

        m.x.fix(0)
        v = list(p.iter_vars())
        self.assertEqual(len(v), 8)

        v = list(p.iter_vars(fixed=False))
        self.assertEqual(len(v), 7)
        self.assertTrue(contains(m.z[3], v))
        self.assertTrue(contains(expr, v))

        v = list(p.iter_vars(fixed=True))
        self.assertEqual(len(v), 1)
        self.assertIn(m.x, v)

        v = list(p.iter_vars(expr_vars=True))
        self.assertEqual(len(v), 10)
        self.assertFalse(contains(expr, v))
        self.assertTrue(contains(m.a, v))
        self.assertTrue(contains(m.b, v))

        m.a.fix(0)
        v = list(p.iter_vars(expr_vars=True, fixed=False))
        self.assertEqual(len(v), 8)

        m.b.fix(0)
        m.c.fix(0)
        v = list(p.iter_vars(expr_vars=True, fixed=False))
        self.assertEqual(len(v), 6)

        v = list(p.iter_vars(fixed=False))
        self.assertEqual(len(v), 6)
        self.assertFalse(contains(expr, v))

        v = list(p.iter_vars(expr_vars=True, names=True))
        self.assertEqual(len(v), 10)
        self.assertEqual(len(v[0]), 3)
        for t in v:
            if t[0] == 'x':
                self.assertIs(t[2], m.x)
                break

    def test_pprint(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Port(implicit=['imp'])
        pipe.OUT.add(-pipe.flow, "flow")
        pipe.OUT.add(pipe.composition, "composition")
        pipe.OUT.add(pipe.composition['a'], "comp_a")
        pipe.OUT.add(pipe.pIn, "pressure")

        os = StringIO()
        pipe.OUT.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""OUT : Size=1, Index=None
    Key  : Name        : Size : Variable
    None :      comp_a :    1 : composition[a]
         : composition :    3 : composition
         :        flow :    1 : - flow
         :         imp :    - : None
         :    pressure :    1 : pIn
""")

        def _IN(m, i):
            return { 'pressure': pipe.pIn,
                     'flow': pipe.composition[i] * pipe.flow }

        pipe.IN = Port(pipe.SPECIES, rule=_IN)
        os = StringIO()
        pipe.IN.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""IN : Size=3, Index=SPECIES
    Key : Name     : Size : Variable
      a :     flow :    1 : composition[a]*flow
        : pressure :    1 :                 pIn
      b :     flow :    1 : composition[b]*flow
        : pressure :    1 :                 pIn
      c :     flow :    1 : composition[c]*flow
        : pressure :    1 :                 pIn
""")
        
    def test_display(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a', 'b', 'c'])
        pipe.flow = Var(initialize=10)
        pipe.composition = Var( pipe.SPECIES,
                                initialize=lambda m,i: ord(i)-ord('a') )
        pipe.pIn  = Var( within=NonNegativeReals, initialize=3.14 )

        pipe.OUT = Port(implicit=['imp'])
        pipe.OUT.add(-pipe.flow, "flow")
        pipe.OUT.add(pipe.composition, "composition")
        pipe.OUT.add(pipe.pIn, "pressure")

        os = StringIO()
        pipe.OUT.display(ostream=os)
        self.assertEqual(os.getvalue(),
"""OUT : Size=1
    Key  : Name        : Value
    None : composition : {'a': 0, 'b': 1, 'c': 2}
         :        flow : -10
         :         imp : -
         :    pressure : 3.14
""")

        def _IN(m, i):
            return { 'pressure': pipe.pIn,
                     'flow': pipe.composition[i] * pipe.flow }

        pipe.IN = Port(pipe.SPECIES, rule=_IN)
        os = StringIO()
        pipe.IN.display(ostream=os)
        self.assertEqual(os.getvalue(),
"""IN : Size=3
    Key : Name     : Value
      a :     flow :     0
        : pressure :  3.14
      b :     flow :    10
        : pressure :  3.14
      c :     flow :    20
        : pressure :  3.14
""")


if __name__ == "__main__":
    unittest.main()
