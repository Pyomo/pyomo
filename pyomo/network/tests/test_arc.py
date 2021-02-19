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
# Unit Tests for Arc
#

import pyutilib.th as unittest
from six import StringIO
import logging

from pyomo.environ import ConcreteModel, AbstractModel, Var, Set, Constraint, RangeSet, NonNegativeReals, Reals, Binary, TransformationFactory, Block
from pyomo.network import Arc, Port

class TestArc(unittest.TestCase):

    def test_default_scalar_constructor(self):
        m = ConcreteModel()
        m.c1 = Arc()
        self.assertEqual(len(m.c1), 0)
        self.assertIsNone(m.c1.directed)
        self.assertIsNone(m.c1.ports)
        self.assertIsNone(m.c1.source)
        self.assertIsNone(m.c1.destination)

        m = AbstractModel()
        m.c1 = Arc()
        self.assertEqual(len(m.c1), 0)
        self.assertIsNone(m.c1.directed)
        self.assertIsNone(m.c1.ports)
        self.assertIsNone(m.c1.source)
        self.assertIsNone(m.c1.destination)

        inst = m.create_instance()
        self.assertEqual(len(inst.c1), 0)
        self.assertIsNone(inst.c1.directed)
        self.assertIsNone(inst.c1.ports)
        self.assertIsNone(inst.c1.source)
        self.assertIsNone(inst.c1.destination)

    def test_default_indexed_constructor(self):
        m = ConcreteModel()
        m.c1 = Arc([1, 2, 3])
        self.assertEqual(len(m.c1), 0)
        self.assertIs(m.c1.ctype, Arc)

        m = AbstractModel()
        m.c1 = Arc([1, 2, 3])
        self.assertEqual(len(m.c1), 0)
        self.assertIs(m.c1.ctype, Arc)


        inst = m.create_instance()
        self.assertEqual(len(m.c1), 0)
        self.assertIs(m.c1.ctype, Arc)

    def test_with_scalar_ports(self):
        def rule(m):
            return dict(source=m.prt1, destination=m.prt2)

        m = ConcreteModel()
        m.prt1 = Port()
        m.prt2 = Port()
        m.c1 = Arc(rule=rule)
        self.assertEqual(len(m.c1), 1)
        self.assertTrue(m.c1.directed)
        self.assertIs(m.c1.source, m.prt1)
        self.assertIs(m.c1.destination, m.prt2)
        self.assertIs(m.c1.ports[0], m.prt1)
        self.assertIs(m.c1.ports[1], m.prt2)
        m.c2 = Arc(ports=(m.prt1, m.prt2))
        self.assertEqual(len(m.c2), 1)
        self.assertFalse(m.c2.directed)
        self.assertIsInstance(m.c2.ports, tuple)
        self.assertEqual(len(m.c2.ports), 2)
        self.assertIs(m.c2.ports[0], m.prt1)
        self.assertIs(m.c2.ports[1], m.prt2)
        self.assertIsNone(m.c2.source)
        self.assertIsNone(m.c2.destination)

        m = AbstractModel()
        m.prt1 = Port()
        m.prt2 = Port()
        m.c1 = Arc(source=m.prt1, destination=m.prt2)
        self.assertEqual(len(m.c1), 0)
        self.assertIsNone(m.c1.directed)
        self.assertIsNone(m.c1.ports)
        self.assertIsNone(m.c1.source)
        self.assertIsNone(m.c1.destination)
        m.c2 = Arc(ports=(m.prt1, m.prt2))
        self.assertEqual(len(m.c2), 0)
        self.assertIsNone(m.c2.directed)
        self.assertIsNone(m.c2.ports)
        self.assertIsNone(m.c2.source)
        self.assertIsNone(m.c2.destination)

        inst = m.create_instance()
        self.assertEqual(len(inst.c1), 1)
        self.assertTrue(inst.c1.directed)
        self.assertIs(inst.c1.source, inst.prt1)
        self.assertIs(inst.c1.destination, inst.prt2)
        self.assertIs(inst.c1.ports[0], inst.prt1)
        self.assertIs(inst.c1.ports[1], inst.prt2)
        self.assertEqual(len(inst.c2), 1)
        self.assertFalse(inst.c2.directed)
        self.assertIsInstance(inst.c2.ports, tuple)
        self.assertEqual(len(inst.c2.ports), 2)
        self.assertIs(inst.c2.ports[0], inst.prt1)
        self.assertIs(inst.c2.ports[1], inst.prt2)
        self.assertIsNone(inst.c2.source)
        self.assertIsNone(inst.c2.destination)

    def test_with_indexed_ports(self):
        def rule1(m, i):
            return dict(source=m.prt1[i], destination=m.prt2[i])
        def rule2(m, i):
            return dict(ports=(m.prt1[i], m.prt2[i]))
        def rule3(m, i):
            # should accept any two-member iterable
            return (c for c in (m.prt1[i], m.prt2[i]))

        m = ConcreteModel()
        m.s = RangeSet(1, 5)
        m.prt1 = Port(m.s)
        m.prt2 = Port(m.s)
        m.c1 = Arc(m.s, rule=rule1)
        self.assertEqual(len(m.c1), 5)
        self.assertTrue(m.c1[4].directed)
        self.assertIs(m.c1[4].source, m.prt1[4])
        self.assertIs(m.c1[4].destination, m.prt2[4])
        self.assertIs(m.c1[4].ports[0], m.prt1[4])
        self.assertIs(m.c1[4].ports[1], m.prt2[4])
        m.c2 = Arc(m.s, rule=rule2)
        self.assertEqual(len(m.c2), 5)
        self.assertFalse(m.c2[4].directed)
        self.assertIsInstance(m.c2[4].ports, tuple)
        self.assertEqual(len(m.c2[4].ports), 2)
        self.assertIs(m.c2[4].ports[0], m.prt1[4])
        self.assertIs(m.c2[4].ports[1], m.prt2[4])
        self.assertIsNone(m.c2[4].source)
        self.assertIsNone(m.c2[4].destination)
        m.c3 = Arc(m.s, rule=rule3, directed=True)
        self.assertEqual(len(m.c3), 5)
        self.assertTrue(m.c3[4].directed)
        self.assertIs(m.c3[4].source, m.prt1[4])
        self.assertIs(m.c3[4].destination, m.prt2[4])
        self.assertIs(m.c3[4].ports[0], m.prt1[4])
        self.assertIs(m.c3[4].ports[1], m.prt2[4])
        m.c4 = Arc(m.s, rule=rule3)
        self.assertEqual(len(m.c4), 5)
        self.assertFalse(m.c4[4].directed)
        self.assertIsInstance(m.c4[4].ports, tuple)
        self.assertEqual(len(m.c4[4].ports), 2)
        self.assertIs(m.c4[4].ports[0], m.prt1[4])
        self.assertIs(m.c4[4].ports[1], m.prt2[4])
        self.assertIsNone(m.c4[4].source)
        self.assertIsNone(m.c4[4].destination)

        logging.disable(logging.ERROR)
        with self.assertRaises(ValueError):
            m.c5 = Arc(m.s, rule=rule1, directed=False)
        logging.disable(logging.NOTSET)

        m = AbstractModel()
        m.s = RangeSet(1, 5)
        m.prt1 = Port(m.s)
        m.prt2 = Port(m.s)
        m.c1 = Arc(m.s, rule=rule1)
        self.assertEqual(len(m.c1), 0)
        self.assertIs(m.c1.ctype, Arc)
        m.c2 = Arc(m.s, rule=rule2)
        self.assertEqual(len(m.c2), 0)
        self.assertIs(m.c1.ctype, Arc)

        inst = m.create_instance()
        self.assertEqual(len(inst.c1), 5)
        self.assertTrue(inst.c1[4].directed)
        self.assertIs(inst.c1[4].source, inst.prt1[4])
        self.assertIs(inst.c1[4].destination, inst.prt2[4])
        self.assertIs(inst.c2[4].ports[0], inst.prt1[4])
        self.assertIs(inst.c2[4].ports[1], inst.prt2[4])
        self.assertEqual(len(inst.c2), 5)
        self.assertFalse(inst.c2[4].directed)
        self.assertIs(inst.c2[4].ports[0], inst.prt1[4])
        self.assertIs(inst.c2[4].ports[1], inst.prt2[4])
        self.assertIsNone(inst.c2[4].source)
        self.assertIsNone(inst.c2[4].destination)

    def test_getattr(self):
        m = ConcreteModel()
        m.x = Var()
        m.p1 = Port(initialize=[m.x])
        m.p2 = Port(extends=m.p1)
        m.a = Arc(ports=(m.p1, m.p2))

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertIs(m.a.x_equality, m.a.expanded_block.x_equality)
        with self.assertRaises(AttributeError):
            m.a.something

    def test_set_value(self):
        m = ConcreteModel()
        m.p1 = Port()
        m.p2 = Port()
        m.p3 = Port()
        m.p4 = Port()
        m.a = Arc(source=m.p1, destination=m.p2)

        self.assertIn(m.a, m.p1.dests())
        self.assertIn(m.a, m.p2.sources())
        self.assertIn(m.a, m.p1.arcs())
        self.assertIn(m.a, m.p2.arcs())

        m.a = dict(ports=(m.p3, m.p4))

        self.assertEqual(len(m.p1.dests()), 0)
        self.assertEqual(len(m.p2.sources()), 0)
        self.assertEqual(len(m.p1.arcs()), 0)
        self.assertEqual(len(m.p2.arcs()), 0)

        self.assertIn(m.a, m.p3.dests())
        self.assertIn(m.a, m.p4.sources())
        self.assertIn(m.a, m.p3.arcs())
        self.assertIn(m.a, m.p4.arcs())

        m.a = dict(ports=(m.p3, m.p4), directed=False)

        self.assertEqual(len(m.p3.dests()), 0)
        self.assertEqual(len(m.p4.sources()), 0)
        self.assertIn(m.a, m.p3.arcs())
        self.assertIn(m.a, m.p4.arcs())

        m.a = (m.p1, m.p2)

        self.assertEqual(len(m.p3.dests()), 0)
        self.assertEqual(len(m.p4.sources()), 0)
        self.assertEqual(len(m.p3.arcs()), 0)
        self.assertEqual(len(m.p4.arcs()), 0)

        self.assertEqual(len(m.p1.dests()), 0)
        self.assertEqual(len(m.p2.sources()), 0)
        self.assertIn(m.a, m.p1.arcs())
        self.assertIn(m.a, m.p2.arcs())

        m.a = dict(ports=(m.p3, m.p4), directed=True)

        self.assertEqual(len(m.p1.dests()), 0)
        self.assertEqual(len(m.p2.sources()), 0)
        self.assertEqual(len(m.p1.arcs()), 0)
        self.assertEqual(len(m.p2.arcs()), 0)

        self.assertIn(m.a, m.p3.dests())
        self.assertIn(m.a, m.p4.sources())
        self.assertIn(m.a, m.p3.arcs())
        self.assertIn(m.a, m.p4.arcs())

        m.a = (m.p1, m.p2)

        self.assertEqual(len(m.p3.dests()), 0)
        self.assertEqual(len(m.p4.sources()), 0)
        self.assertEqual(len(m.p3.arcs()), 0)
        self.assertEqual(len(m.p4.arcs()), 0)

        self.assertIn(m.a, m.p1.dests())
        self.assertIn(m.a, m.p2.sources())
        self.assertIn(m.a, m.p1.arcs())
        self.assertIn(m.a, m.p2.arcs())

    def test_pprint(self):
        m = ConcreteModel()
        m.s = RangeSet(1, 5)
        m.prt1 = Port(m.s)
        m.prt2 = Port(m.s)

        @m.Arc(m.s)
        def friend(m, i):
            return dict(source=m.prt1[i], destination=m.prt2[i])

        os = StringIO()
        m.friend.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""friend : Size=5, Index=s, Active=True
    Key : Ports              : Directed : Active
      1 : (prt1[1], prt2[1]) :     True :   True
      2 : (prt1[2], prt2[2]) :     True :   True
      3 : (prt1[3], prt2[3]) :     True :   True
      4 : (prt1[4], prt2[4]) :     True :   True
      5 : (prt1[5], prt2[5]) :     True :   True
""")

        m = ConcreteModel()
        m.z = RangeSet(1, 2)
        m.prt1 = Port(m.z)
        m.prt2 = Port(m.z)

        @m.Arc(m.z)
        def pal(m, i):
            return (m.prt1[i], m.prt2[i])

        m.pal[2].deactivate()

        os = StringIO()
        m.pal.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""pal : Size=2, Index=z, Active=True
    Key : Ports              : Directed : Active
      1 : (prt1[1], prt2[1]) :    False :   True
      2 : (prt1[2], prt2[2]) :    False :  False
""")


    def test_expand_single_scalar(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.prt1 = Port()
        m.prt1.add(m.x, "v")
        m.prt2 = Port()
        m.prt2.add(m.y, "v")

        # Both should be expanded, but for d all we get is an empty block
        m.c = Arc(source=m.prt1, destination=m.prt2)
        m.d = Arc()

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 1)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 1)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.active)
        self.assertTrue(blk.component('v_equality').active)
        self.assertTrue(m.d_expanded.active)
        self.assertEqual(len(list(m.d_expanded.component_objects())), 0)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    1 Constraint Declarations
        v_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :   0.0 : x - y :   0.0 :   True

    1 Declarations: v_equality
""")


    def test_expand_scalar(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.w = Var()
        m.prt1 = Port()
        m.prt1.add(m.x, "a")
        m.prt1.add(m.y, "b")
        m.prt2 = Port()
        m.prt2.add(m.z, "a")
        m.prt2.add(m.w, "b")

        m.c = Arc(ports=(m.prt1, m.prt2))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.active)
        self.assertTrue(blk.component('a_equality').active)
        self.assertTrue(blk.component('b_equality').active)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        a_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :   0.0 : x - z :   0.0 :   True
        b_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :   0.0 : y - w :   0.0 :   True

    2 Declarations: a_equality b_equality
""")


    def test_expand_expression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.w = Var()
        m.prt1 = Port()
        m.prt1.add(-m.x, name='expr1')
        m.prt1.add(1 + m.y, name='expr2')
        m.prt2 = Port()
        m.prt2.add(-m.z, name='expr1')
        m.prt2.add(1 + m.w, name='expr2')

        m.c = Arc(ports=(m.prt1, m.prt2))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.active)
        self.assertTrue(blk.component('expr1_equality').active)
        self.assertTrue(blk.component('expr2_equality').active)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        expr1_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body    : Upper : Active
            None :   0.0 : - x + z :   0.0 :   True
        expr2_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body            : Upper : Active
            None :   0.0 : 1 + y - (1 + w) :   0.0 :   True

    2 Declarations: expr1_equality expr2_equality
""")


    def test_expand_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2])
        m.y = Var([1,2], [1,2])
        m.z = Var()
        m.t = Var([1,2])
        m.u = Var([1,2], [1,2])
        m.v = Var()
        m.prt1 = Port()
        m.prt1.add(m.x, "a")
        m.prt1.add(m.y, "b")
        m.prt1.add(m.z, "c")
        m.prt2 = Port()
        m.prt2.add(m.t, "a")
        m.prt2.add(m.u, "b")
        m.prt2.add(m.v, "c")

        m.c = Arc(ports=(m.prt1, m.prt2))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 7)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.active)
        self.assertTrue(blk.component('a_equality').active)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    3 Constraint Declarations
        a_equality : Size=2, Index=x_index, Active=True
            Key : Lower : Body        : Upper : Active
              1 :   0.0 : x[1] - t[1] :   0.0 :   True
              2 :   0.0 : x[2] - t[2] :   0.0 :   True
        b_equality : Size=4, Index=y_index, Active=True
            Key    : Lower : Body            : Upper : Active
            (1, 1) :   0.0 : y[1,1] - u[1,1] :   0.0 :   True
            (1, 2) :   0.0 : y[1,2] - u[1,2] :   0.0 :   True
            (2, 1) :   0.0 : y[2,1] - u[2,1] :   0.0 :   True
            (2, 2) :   0.0 : y[2,2] - u[2,2] :   0.0 :   True
        c_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :   0.0 : z - v :   0.0 :   True

    3 Declarations: a_equality b_equality c_equality
""")


    def test_expand_trivial(self):
        m = ConcreteModel()
        m.x = Var()
        m.prt = Port()
        m.prt.add(m.x, "a")

        m.c = Arc(ports=(m.prt, m.prt))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 1)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 1)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.active)
        self.assertTrue(blk.component('a_equality').active)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    1 Constraint Declarations
        a_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :   0.0 : x - x :   0.0 :   True

    1 Declarations: a_equality
""")


    def test_expand_empty_scalar(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1,3))
        m.y = Var(domain=Binary)
        m.PRT = Port()
        m.PRT.add(m.x)
        m.PRT.add(m.y)
        m.EPRT = Port()

        m.c = Arc(ports=(m.PRT, m.EPRT))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.component('x_equality').active)
        self.assertTrue(blk.component('y_equality').active)

        self.assertIs( m.x.domain, m.component('EPRT_auto_x').domain )
        self.assertIs( m.y.domain, m.component('EPRT_auto_y').domain )
        self.assertEqual( m.x.bounds, m.component('EPRT_auto_x').bounds )
        self.assertEqual( m.y.bounds, m.component('EPRT_auto_y').bounds )

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body            : Upper : Active
            None :   0.0 : x - EPRT_auto_x :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body            : Upper : Active
            None :   0.0 : y - EPRT_auto_y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")


    def test_expand_empty_expression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.PRT = Port()
        m.PRT.add(-m.x, 'x')
        m.PRT.add(1 + m.y, 'y')
        m.EPRT = Port()

        m.c = Arc(ports=(m.PRT, m.EPRT))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.component('x_equality').active)
        self.assertTrue(blk.component('y_equality').active)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body              : Upper : Active
            None :   0.0 : - x - EPRT_auto_x :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body                : Upper : Active
            None :   0.0 : 1 + y - EPRT_auto_y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")


    def test_expand_empty_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.PRT = Port()
        m.PRT.add(m.x)
        m.PRT.add(m.y)
        m.EPRT = Port()

        m.c = Arc(ports=(m.PRT, m.EPRT))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 3)
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.component('x_equality').active)
        self.assertTrue(blk.component('y_equality').active)

        self.assertIs( m.x[1].domain, m.component('EPRT_auto_x')[1].domain )
        self.assertIs( m.x[2].domain, m.component('EPRT_auto_x')[2].domain )
        self.assertIs( m.y.domain, m.component('EPRT_auto_y').domain )
        self.assertEqual( m.x[1].bounds, m.component('EPRT_auto_x')[1].bounds )
        self.assertEqual( m.x[2].bounds, m.component('EPRT_auto_x')[2].bounds )
        self.assertEqual( m.y.bounds, m.component('EPRT_auto_y').bounds )

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=x_index, Active=True
            Key : Lower : Body                  : Upper : Active
              1 :   0.0 : x[1] - EPRT_auto_x[1] :   0.0 :   True
              2 :   0.0 : x[2] - EPRT_auto_x[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body            : Upper : Active
            None :   0.0 : y - EPRT_auto_y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")


    def test_expand_multiple_empty_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.PRT = Port()
        m.PRT.add(m.x)
        m.PRT.add(m.y)
        m.EPRT2 = Port()
        m.EPRT1 = Port()

        # Define d first to test that it knows how to expand the EPRTs
        m.d = Arc(ports=(m.EPRT2, m.EPRT1))
        m.c = Arc(ports=(m.PRT, m.EPRT1))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 4)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 6)
        self.assertFalse(m.c.active)
        blk_c = m.component('c_expanded')
        self.assertTrue(blk_c.component('x_equality').active)
        self.assertTrue(blk_c.component('y_equality').active)
        self.assertFalse(m.d.active)
        blk_d = m.component('d_expanded')
        self.assertTrue(blk_d.component('x_equality').active)
        self.assertTrue(blk_d.component('y_equality').active)

        self.assertIs( m.x[1].domain, m.component('EPRT1_auto_x')[1].domain )
        self.assertIs( m.x[2].domain, m.component('EPRT1_auto_x')[2].domain )
        self.assertIs( m.y.domain, m.component('EPRT1_auto_y').domain )
        self.assertEqual( m.x[1].bounds, m.component('EPRT1_auto_x')[1].bounds )
        self.assertEqual( m.x[2].bounds, m.component('EPRT1_auto_x')[2].bounds )
        self.assertEqual( m.y.bounds, m.component('EPRT1_auto_y').bounds )

        self.assertIs( m.x[1].domain, m.component('EPRT2_auto_x')[1].domain )
        self.assertIs( m.x[2].domain, m.component('EPRT2_auto_x')[2].domain )
        self.assertIs( m.y.domain, m.component('EPRT2_auto_y').domain )
        self.assertEqual( m.x[1].bounds, m.component('EPRT2_auto_x')[1].bounds )
        self.assertEqual( m.x[2].bounds, m.component('EPRT2_auto_x')[2].bounds )
        self.assertEqual( m.y.bounds, m.component('EPRT2_auto_y').bounds )

        os = StringIO()
        blk_c.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=x_index, Active=True
            Key : Lower : Body                   : Upper : Active
              1 :   0.0 : x[1] - EPRT1_auto_x[1] :   0.0 :   True
              2 :   0.0 : x[2] - EPRT1_auto_x[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body             : Upper : Active
            None :   0.0 : y - EPRT1_auto_y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")

        os = StringIO()
        blk_d.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=x_index, Active=True
            Key : Lower : Body                              : Upper : Active
              1 :   0.0 : EPRT2_auto_x[1] - EPRT1_auto_x[1] :   0.0 :   True
              2 :   0.0 : EPRT2_auto_x[2] - EPRT1_auto_x[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body                        : Upper : Active
            None :   0.0 : EPRT2_auto_y - EPRT1_auto_y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")


    def test_expand_multiple_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.PRT = Port()
        m.PRT.add(m.x)
        m.PRT.add(m.y)
        m.a1 = Var([1,2])
        m.a2 = Var([1,2])
        m.b1 = Var()
        m.b2 = Var()
        m.EPRT1 = Port()
        m.EPRT1.add(m.a1,'x')
        m.EPRT1.add(m.b1,'y')
        m.EPRT2 = Port()
        m.EPRT2.add(m.a2,'x')
        m.EPRT2.add(m.b2,'y')

        m.c = Arc(ports=(m.PRT, m.EPRT1))
        m.d = Arc(ports=(m.EPRT2, m.EPRT1))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 4)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 6)
        self.assertFalse(m.c.active)
        blk_c = m.component('c_expanded')
        self.assertTrue(blk_c.component('x_equality').active)
        self.assertTrue(blk_c.component('y_equality').active)
        self.assertFalse(m.d.active)
        blk_d = m.component('d_expanded')
        self.assertTrue(blk_d.component('x_equality').active)
        self.assertTrue(blk_d.component('y_equality').active)

        os = StringIO()
        blk_c.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=x_index, Active=True
            Key : Lower : Body         : Upper : Active
              1 :   0.0 : x[1] - a1[1] :   0.0 :   True
              2 :   0.0 : x[2] - a1[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body   : Upper : Active
            None :   0.0 : y - b1 :   0.0 :   True

    2 Declarations: x_equality y_equality
""")

        os = StringIO()
        blk_d.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=x_index, Active=True
            Key : Lower : Body          : Upper : Active
              1 :   0.0 : a2[1] - a1[1] :   0.0 :   True
              2 :   0.0 : a2[2] - a1[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body    : Upper : Active
            None :   0.0 : b2 - b1 :   0.0 :   True

    2 Declarations: x_equality y_equality
""")


    def test_expand_implicit_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.PRT = Port()
        m.PRT.add(m.x)
        m.PRT.add(m.y)
        m.a2 = Var([1,2])
        m.b1 = Var()
        m.EPRT2 = Port(implicit=['x'])
        m.EPRT2.add(m.b1,'y')
        m.EPRT1 = Port(implicit=['y'])
        m.EPRT1.add(m.a2,'x')

        m.c = Arc(ports=(m.EPRT1, m.PRT))
        m.d = Arc(ports=(m.EPRT2, m.PRT))

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        os = StringIO()
        m.EPRT1.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""EPRT1 : Size=1, Index=None
    Key  : Name : Size : Variable
    None :    x :    2 :       a2
         :    y :    - :     None
""")

        TransformationFactory('network.expand_arcs').apply_to(m)

        os = StringIO()
        m.EPRT1.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""EPRT1 : Size=1, Index=None
    Key  : Name : Size : Variable
    None :    x :    2 :           a2
         :    y :    1 : EPRT1_auto_y
""")

        self.assertEqual(len(list(m.component_objects(Constraint))), 4)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 6)
        self.assertFalse(m.c.active)
        blk_c = m.component('c_expanded')
        self.assertTrue(blk_c.component('x_equality').active)
        self.assertTrue(blk_c.component('y_equality').active)
        self.assertFalse(m.d.active)
        blk_d = m.component('d_expanded')
        self.assertTrue(blk_d.component('x_equality').active)
        self.assertTrue(blk_d.component('y_equality').active)

        os = StringIO()
        blk_c.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=a2_index, Active=True
            Key : Lower : Body         : Upper : Active
              1 :   0.0 : a2[1] - x[1] :   0.0 :   True
              2 :   0.0 : a2[2] - x[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body             : Upper : Active
            None :   0.0 : EPRT1_auto_y - y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")

        os = StringIO()
        blk_d.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d_expanded : Size=1, Index=None, Active=True
    2 Constraint Declarations
        x_equality : Size=2, Index=a2_index, Active=True
            Key : Lower : Body                   : Upper : Active
              1 :   0.0 : EPRT2_auto_x[1] - x[1] :   0.0 :   True
              2 :   0.0 : EPRT2_auto_x[2] - x[2] :   0.0 :   True
        y_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body   : Upper : Active
            None :   0.0 : b1 - y :   0.0 :   True

    2 Declarations: x_equality y_equality
""")


    def test_expand_indexed_arc(self):
        def rule(m, i):
            return (m.c1[i], m.c2[i])

        m = ConcreteModel()
        m.x = Var(initialize=1, domain=Reals)
        m.y = Var(initialize=2, domain=Reals)
        m.c1 = Port([1, 2])
        m.c1[1].add(m.x, name='v')
        m.c1[2].add(m.y, name='t')
        m.z = Var(initialize=1, domain=Reals)
        m.w = Var(initialize=2, domain=Reals)
        m.c2 = Port([1, 2])
        m.c2[1].add(m.z, name='v')
        m.c2[2].add(m.w, name='t')

        m.eq = Arc([1, 2], rule=rule)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertFalse(m.eq.active)
        self.assertFalse(m.eq[1].active)
        self.assertFalse(m.eq[2].active)
        self.assertIs(m.eq.expanded_block, m.eq_expanded)
        self.assertIs(m.eq.expanded_block[1], m.eq_expanded[1])
        self.assertIs(m.eq.expanded_block[2], m.eq_expanded[2])

        os = StringIO()
        m.component('eq_expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""eq_expanded : Size=2, Index=eq_index, Active=True
    eq_expanded[1] : Active=True
        1 Constraint Declarations
            v_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body  : Upper : Active
                None :   0.0 : x - z :   0.0 :   True

        1 Declarations: v_equality
    eq_expanded[2] : Active=True
        1 Constraint Declarations
            t_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body  : Upper : Active
                None :   0.0 : y - w :   0.0 :   True

        1 Declarations: t_equality
""")


    def test_inactive(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.prt1 = Port()
        m.prt1.add(m.x, "v")
        m.prt2 = Port()
        m.prt2.add(m.y, "v")
        m.prt3 = Port()
        m.prt3.add(m.z, "v")

        # The active arc should be deactivated and expanded,
        # but not the inactive Arc
        m.c = Arc(source=m.prt1, destination=m.prt2)
        m.inactive = Arc(ports=(m.prt3, m.prt2))
        m.inactive.deactivate()

        self.assertEqual(len(list(m.component_objects(Constraint))), 0)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 0)

        TransformationFactory('network.expand_arcs').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 1)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 1)
        self.assertFalse(m.inactive.active)
        self.assertIsNone(m.component('inactive_expanded'))
        self.assertFalse(m.c.active)
        blk = m.component('c_expanded')
        self.assertTrue(blk.active)
        self.assertTrue(blk.component('v_equality').active)

        os = StringIO()
        blk.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c_expanded : Size=1, Index=None, Active=True
    1 Constraint Declarations
        v_equality : Size=1, Index=None, Active=True
            Key  : Lower : Body  : Upper : Active
            None :   0.0 : x - y :   0.0 :   True

    1 Declarations: v_equality
""")

    def test_extensive_no_splitfrac_single_var(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.p1 = Port(initialize={'v': (m.x, Port.Extensive, {'include_splitfrac':False})})
        m.p2 = Port(initialize={'v': (m.y, Port.Extensive, {'include_splitfrac':False})})
        m.p3 = Port(initialize={'v': (m.z, Port.Extensive, {'include_splitfrac':False})})
        m.a1 = Arc(source=m.p1, destination=m.p2)
        m.a2 = Arc(source=m.p1, destination=m.p3)

        TransformationFactory('network.expand_arcs').apply_to(m)

        os = StringIO()
        m.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""3 Var Declarations
    x : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    y : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    z : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals

3 Constraint Declarations
    p1_v_outsum : Size=1, Index=None, Active=True
        Key  : Lower : Body                              : Upper : Active
        None :   0.0 : a1_expanded.v + a2_expanded.v - x :   0.0 :   True
    p2_v_insum : Size=1, Index=None, Active=True
        Key  : Lower : Body              : Upper : Active
        None :   0.0 : a1_expanded.v - y :   0.0 :   True
    p3_v_insum : Size=1, Index=None, Active=True
        Key  : Lower : Body              : Upper : Active
        None :   0.0 : a2_expanded.v - z :   0.0 :   True

2 Block Declarations
    a1_expanded : Size=1, Index=None, Active=True
        1 Var Declarations
            v : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        1 Declarations: v
    a2_expanded : Size=1, Index=None, Active=True
        1 Var Declarations
            v : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        1 Declarations: v

2 Arc Declarations
    a1 : Size=1, Index=None, Active=False
        Key  : Ports    : Directed : Active
        None : (p1, p2) :     True :  False
    a2 : Size=1, Index=None, Active=False
        Key  : Ports    : Directed : Active
        None : (p1, p3) :     True :  False

3 Port Declarations
    p1 : Size=1, Index=None
        Key  : Name : Size : Variable
        None :    v :    1 :        x
    p2 : Size=1, Index=None
        Key  : Name : Size : Variable
        None :    v :    1 :        y
    p3 : Size=1, Index=None
        Key  : Name : Size : Variable
        None :    v :    1 :        z

13 Declarations: x y z p1 p2 p3 a1 a2 a1_expanded a2_expanded p1_v_outsum p2_v_insum p3_v_insum
""")

    def test_extensive_single_var(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.p1 = Port(initialize={'v': (m.x, Port.Extensive)})
        m.p2 = Port(initialize={'v': (m.y, Port.Extensive)})
        m.p3 = Port(initialize={'v': (m.z, Port.Extensive)})
        m.a1 = Arc(source=m.p1, destination=m.p2)
        m.a2 = Arc(source=m.p1, destination=m.p3)

        TransformationFactory('network.expand_arcs').apply_to(m)

        os = StringIO()
        m.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""3 Var Declarations
    x : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    y : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals
    z : Size=1, Index=None
        Key  : Lower : Value : Upper : Fixed : Stale : Domain
        None :  None :  None :  None : False :  True :  Reals

3 Constraint Declarations
    p1_v_outsum : Size=1, Index=None, Active=True
        Key  : Lower : Body                              : Upper : Active
        None :   0.0 : a1_expanded.v + a2_expanded.v - x :   0.0 :   True
    p2_v_insum : Size=1, Index=None, Active=True
        Key  : Lower : Body              : Upper : Active
        None :   0.0 : a1_expanded.v - y :   0.0 :   True
    p3_v_insum : Size=1, Index=None, Active=True
        Key  : Lower : Body              : Upper : Active
        None :   0.0 : a2_expanded.v - z :   0.0 :   True

2 Block Declarations
    a1_expanded : Size=1, Index=None, Active=True
        1 Var Declarations
            v : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        1 Declarations: v
    a2_expanded : Size=1, Index=None, Active=True
        1 Var Declarations
            v : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        1 Declarations: v

2 Arc Declarations
    a1 : Size=1, Index=None, Active=False
        Key  : Ports    : Directed : Active
        None : (p1, p2) :     True :  False
    a2 : Size=1, Index=None, Active=False
        Key  : Ports    : Directed : Active
        None : (p1, p3) :     True :  False

3 Port Declarations
    p1 : Size=1, Index=None
        Key  : Name : Size : Variable
        None :    v :    1 :        x
    p2 : Size=1, Index=None
        Key  : Name : Size : Variable
        None :    v :    1 :        y
    p3 : Size=1, Index=None
        Key  : Name : Size : Variable
        None :    v :    1 :        z

13 Declarations: x y z p1 p2 p3 a1 a2 a1_expanded a2_expanded p1_v_outsum p2_v_insum p3_v_insum
""")

    def test_extensive_no_splitfrac_expansion(self):
        m = ConcreteModel()
        m.time = Set(initialize=[1, 2, 3])

        m.source = Block()
        m.load1 = Block()
        m.load2 = Block()

        def source_block(b):
            b.p_out = Var(b.model().time)
            b.outlet = Port(initialize={'p': (b.p_out, Port.Extensive, {'include_splitfrac':False})})

        def load_block(b):
            b.p_in = Var(b.model().time)
            b.inlet = Port(initialize={'p': (b.p_in, Port.Extensive, {'include_splitfrac':False})})

        source_block(m.source)
        load_block(m.load1)
        load_block(m.load2)

        m.cs1 = Arc(source=m.source.outlet, destination=m.load1.inlet)
        m.cs2 = Arc(source=m.source.outlet, destination=m.load2.inlet)

        TransformationFactory("network.expand_arcs").apply_to(m)

        ref = """
1 Set Declarations
    time : Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :    3 : {1, 2, 3}

5 Block Declarations
    cs1_expanded : Size=1, Index=None, Active=True
        1 Var Declarations
            p : Size=3, Index=time
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  1 :  None :  None :  None : False :  True :  Reals
                  2 :  None :  None :  None : False :  True :  Reals
                  3 :  None :  None :  None : False :  True :  Reals

        1 Declarations: p
    cs2_expanded : Size=1, Index=None, Active=True
        1 Var Declarations
            p : Size=3, Index=time
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  1 :  None :  None :  None : False :  True :  Reals
                  2 :  None :  None :  None : False :  True :  Reals
                  3 :  None :  None :  None : False :  True :  Reals

        1 Declarations: p
    load1 : Size=1, Index=None, Active=True
        1 Var Declarations
            p_in : Size=3, Index=time
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  1 :  None :  None :  None : False :  True :  Reals
                  2 :  None :  None :  None : False :  True :  Reals
                  3 :  None :  None :  None : False :  True :  Reals

        1 Constraint Declarations
            inlet_p_insum : Size=3, Index=time, Active=True
                Key : Lower : Body                              : Upper : Active
                  1 :   0.0 : cs1_expanded.p[1] - load1.p_in[1] :   0.0 :   True
                  2 :   0.0 : cs1_expanded.p[2] - load1.p_in[2] :   0.0 :   True
                  3 :   0.0 : cs1_expanded.p[3] - load1.p_in[3] :   0.0 :   True

        1 Port Declarations
            inlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None :    p :    3 : load1.p_in

        3 Declarations: p_in inlet inlet_p_insum
    load2 : Size=1, Index=None, Active=True
        1 Var Declarations
            p_in : Size=3, Index=time
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  1 :  None :  None :  None : False :  True :  Reals
                  2 :  None :  None :  None : False :  True :  Reals
                  3 :  None :  None :  None : False :  True :  Reals

        1 Constraint Declarations
            inlet_p_insum : Size=3, Index=time, Active=True
                Key : Lower : Body                              : Upper : Active
                  1 :   0.0 : cs2_expanded.p[1] - load2.p_in[1] :   0.0 :   True
                  2 :   0.0 : cs2_expanded.p[2] - load2.p_in[2] :   0.0 :   True
                  3 :   0.0 : cs2_expanded.p[3] - load2.p_in[3] :   0.0 :   True

        1 Port Declarations
            inlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None :    p :    3 : load2.p_in

        3 Declarations: p_in inlet inlet_p_insum
    source : Size=1, Index=None, Active=True
        1 Var Declarations
            p_out : Size=3, Index=time
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  1 :  None :  None :  None : False :  True :  Reals
                  2 :  None :  None :  None : False :  True :  Reals
                  3 :  None :  None :  None : False :  True :  Reals

        1 Constraint Declarations
            outlet_p_outsum : Size=3, Index=time, Active=True
                Key : Lower : Body                                                    : Upper : Active
                  1 :   0.0 : cs1_expanded.p[1] + cs2_expanded.p[1] - source.p_out[1] :   0.0 :   True
                  2 :   0.0 : cs1_expanded.p[2] + cs2_expanded.p[2] - source.p_out[2] :   0.0 :   True
                  3 :   0.0 : cs1_expanded.p[3] + cs2_expanded.p[3] - source.p_out[3] :   0.0 :   True

        1 Port Declarations
            outlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None :    p :    3 : source.p_out

        3 Declarations: p_out outlet outlet_p_outsum

2 Arc Declarations
    cs1 : Size=1, Index=None, Active=False
        Key  : Ports                        : Directed : Active
        None : (source.outlet, load1.inlet) :     True :  False
    cs2 : Size=1, Index=None, Active=False
        Key  : Ports                        : Directed : Active
        None : (source.outlet, load2.inlet) :     True :  False

8 Declarations: time source load1 load2 cs1 cs2 cs1_expanded cs2_expanded
"""
        os = StringIO()
        m.pprint(ostream=os)
        self.assertEqual(os.getvalue().strip(), ref.strip())

    def test_extensive_expansion(self):
        m = ConcreteModel()
        m.comp = Set(initialize=["a", "b", "c"])

        # Feed
        m.feed = Block()
        m.feed.flow = Var(m.comp, domain=NonNegativeReals)
        m.feed.mass = Var()
        m.feed.temp = Var()

        m.feed.outlet = Port()
        m.feed.outlet.add(m.feed.flow, "flow", rule=Port.Extensive, write_var_sum=False)
        m.feed.outlet.add(m.feed.mass, "mass", rule=Port.Extensive)
        m.feed.outlet.add(m.feed.temp, "temp")

        # Treatment Unit
        m.tru = Block()
        m.tru.flow_in = Var(m.comp, domain=NonNegativeReals)
        m.tru.flow_out = Var(m.comp, domain=NonNegativeReals)
        m.tru.mass = Var()
        m.tru.temp = Var()

        m.tru.inlet = Port()
        m.tru.inlet.add(m.tru.flow_in, "flow", rule=Port.Extensive)
        m.tru.inlet.add(m.tru.mass, "mass", rule=Port.Extensive)
        m.tru.inlet.add(m.tru.temp, "temp")

        m.tru.outlet = Port()
        m.tru.outlet.add(m.tru.flow_out, "flow", rule=Port.Extensive)
        m.tru.outlet.add(m.tru.mass, "mass", rule=Port.Extensive)
        m.tru.outlet.add(m.tru.temp, "temp")

        # Ports with both in and out, connected to each other 1-to-1
        m.node1 = Block()
        m.node1.flow = Var(m.comp, domain=NonNegativeReals)
        m.node1.mass = Var()
        m.node1.temp = Var()

        m.node1.port = Port(initialize=[(m.node1.flow, Port.Extensive),
                                        (m.node1.mass, Port.Extensive),
                                        m.node1.temp])

        m.node2 = Block()
        m.node2.flow = Var(m.comp, domain=NonNegativeReals)
        m.node2.mass = Var()
        m.node2.temp = Var()

        m.node2.port = Port(initialize=[(m.node2.flow, Port.Extensive),
                                        (m.node2.mass, Port.Extensive),
                                        m.node2.temp])

        # Port with multiple inlets and outlets
        m.multi = Block()
        m.multi.flow = Var(m.comp, domain=NonNegativeReals)
        m.multi.mass = Var()
        m.multi.temp = Var()

        m.multi.port = Port(initialize=[(m.multi.flow, Port.Extensive),
                                        (m.multi.mass, Port.Extensive),
                                        m.multi.temp])

        # Product
        m.prod = Block()
        m.prod.flow = Var(m.comp, domain=NonNegativeReals)
        m.prod.mass = Var()
        m.prod.temp = Var()

        m.prod.inlet = Port()
        m.prod.inlet.add(m.prod.flow, "flow", rule=Port.Extensive)
        m.prod.inlet.add(m.prod.mass, "mass", rule=Port.Extensive)
        m.prod.inlet.add(m.prod.temp, "temp")

        # Arcs
        m.stream0 = Arc(source=m.tru.outlet, destination=m.node1.port)
        m.stream1 = Arc(source=m.feed.outlet, destination=m.tru.inlet)
        m.stream2 = Arc(source=m.feed.outlet, destination=m.prod.inlet)
        m.stream3 = Arc(source=m.feed.outlet, destination=m.node1.port)
        m.stream4 = Arc(source=m.node1.port, destination=m.node2.port)
        m.stream5 = Arc(source=m.tru.outlet, destination=m.prod.inlet)
        m.stream6 = Arc(source=m.node2.port, destination=m.prod.inlet)
        m.stream7 = Arc(source=m.feed.outlet, destination=m.multi.port)
        m.stream8 = Arc(source=m.tru.outlet, destination=m.multi.port)
        m.stream9 = Arc(source=m.multi.port, destination=m.prod.inlet)
        m.stream10 = Arc(source=m.multi.port, destination=m.tru.inlet)

        # SplitFrac specifications
        m.feed.outlet.set_split_fraction(m.stream1, .6, fix=True)

        m.stream0.deactivate()

        TransformationFactory('network.expand_arcs').apply_to(m)

        os = StringIO()
        m.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""1 Set Declarations
    comp : Size=1, Index=None, Ordered=Insertion
        Key  : Dimen : Domain : Size : Members
        None :     1 :    Any :    3 : {'a', 'b', 'c'}

16 Block Declarations
    feed : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            temp : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        2 Constraint Declarations
            outlet_frac_sum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                                                                              : Upper : Active
                None :   1.0 : stream1_expanded.splitfrac + stream2_expanded.splitfrac + stream3_expanded.splitfrac + stream7_expanded.splitfrac :   1.0 :   True
            outlet_mass_outsum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                                                                      : Upper : Active
                None :   0.0 : stream1_expanded.mass + stream2_expanded.mass + stream3_expanded.mass + stream7_expanded.mass - feed.mass :   0.0 :   True

        1 Port Declarations
            outlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : feed.flow
                     : mass :    1 : feed.mass
                     : temp :    1 : feed.temp

        6 Declarations: flow mass temp outlet outlet_frac_sum outlet_mass_outsum
    multi : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            temp : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        4 Constraint Declarations
            port_flow_insum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                : Upper : Active
                  a :   0.0 : stream7_expanded.flow[a] + stream8_expanded.flow[a] - multi.flow[a] :   0.0 :   True
                  b :   0.0 : stream7_expanded.flow[b] + stream8_expanded.flow[b] - multi.flow[b] :   0.0 :   True
                  c :   0.0 : stream7_expanded.flow[c] + stream8_expanded.flow[c] - multi.flow[c] :   0.0 :   True
            port_flow_outsum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                 : Upper : Active
                  a :   0.0 : stream9_expanded.flow[a] + stream10_expanded.flow[a] - multi.flow[a] :   0.0 :   True
                  b :   0.0 : stream9_expanded.flow[b] + stream10_expanded.flow[b] - multi.flow[b] :   0.0 :   True
                  c :   0.0 : stream9_expanded.flow[c] + stream10_expanded.flow[c] - multi.flow[c] :   0.0 :   True
            port_mass_insum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                       : Upper : Active
                None :   0.0 : stream7_expanded.mass + stream8_expanded.mass - multi.mass :   0.0 :   True
            port_mass_outsum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                        : Upper : Active
                None :   0.0 : stream9_expanded.mass + stream10_expanded.mass - multi.mass :   0.0 :   True

        1 Port Declarations
            port : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : multi.flow
                     : mass :    1 : multi.mass
                     : temp :    1 : multi.temp

        8 Declarations: flow mass temp port port_flow_outsum port_flow_insum port_mass_outsum port_mass_insum
    node1 : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            temp : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        2 Constraint Declarations
            port_flow_insum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                     : Upper : Active
                  a :   0.0 : stream3_expanded.flow[a] - node1.flow[a] :   0.0 :   True
                  b :   0.0 : stream3_expanded.flow[b] - node1.flow[b] :   0.0 :   True
                  c :   0.0 : stream3_expanded.flow[c] - node1.flow[c] :   0.0 :   True
            port_mass_insum : Size=1, Index=None, Active=True
                Key  : Lower : Body                               : Upper : Active
                None :   0.0 : stream3_expanded.mass - node1.mass :   0.0 :   True

        1 Port Declarations
            port : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : node1.flow
                     : mass :    1 : node1.mass
                     : temp :    1 : node1.temp

        6 Declarations: flow mass temp port port_flow_insum port_mass_insum
    node2 : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            temp : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        2 Constraint Declarations
            port_flow_outsum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                     : Upper : Active
                  a :   0.0 : stream6_expanded.flow[a] - node2.flow[a] :   0.0 :   True
                  b :   0.0 : stream6_expanded.flow[b] - node2.flow[b] :   0.0 :   True
                  c :   0.0 : stream6_expanded.flow[c] - node2.flow[c] :   0.0 :   True
            port_mass_outsum : Size=1, Index=None, Active=True
                Key  : Lower : Body                               : Upper : Active
                None :   0.0 : stream6_expanded.mass - node2.mass :   0.0 :   True

        1 Port Declarations
            port : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : node2.flow
                     : mass :    1 : node2.mass
                     : temp :    1 : node2.temp

        6 Declarations: flow mass temp port port_flow_outsum port_mass_outsum
    prod : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            temp : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        2 Constraint Declarations
            inlet_flow_insum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                                                                     : Upper : Active
                  a :   0.0 : stream2_expanded.flow[a] + stream5_expanded.flow[a] + stream6_expanded.flow[a] + stream9_expanded.flow[a] - prod.flow[a] :   0.0 :   True
                  b :   0.0 : stream2_expanded.flow[b] + stream5_expanded.flow[b] + stream6_expanded.flow[b] + stream9_expanded.flow[b] - prod.flow[b] :   0.0 :   True
                  c :   0.0 : stream2_expanded.flow[c] + stream5_expanded.flow[c] + stream6_expanded.flow[c] + stream9_expanded.flow[c] - prod.flow[c] :   0.0 :   True
            inlet_mass_insum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                                                                      : Upper : Active
                None :   0.0 : stream2_expanded.mass + stream5_expanded.mass + stream6_expanded.mass + stream9_expanded.mass - prod.mass :   0.0 :   True

        1 Port Declarations
            inlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : prod.flow
                     : mass :    1 : prod.mass
                     : temp :    1 : prod.temp

        6 Declarations: flow mass temp inlet inlet_flow_insum inlet_mass_insum
    stream10_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                  : Upper : Active
                  a :   0.0 : stream10_expanded.flow[a] - stream10_expanded.splitfrac*multi.flow[a] :   0.0 :   True
                  b :   0.0 : stream10_expanded.flow[b] - stream10_expanded.splitfrac*multi.flow[b] :   0.0 :   True
                  c :   0.0 : stream10_expanded.flow[c] - stream10_expanded.splitfrac*multi.flow[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                            : Upper : Active
                None :   0.0 : stream10_expanded.mass - stream10_expanded.splitfrac*multi.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                  : Upper : Active
                None :   0.0 : multi.temp - tru.temp :   0.0 :   True

        6 Declarations: flow mass temp_equality splitfrac flow_split mass_split
    stream1_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :   0.6 :  None :  True : False :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                               : Upper : Active
                  a :   0.0 : stream1_expanded.flow[a] - stream1_expanded.splitfrac*feed.flow[a] :   0.0 :   True
                  b :   0.0 : stream1_expanded.flow[b] - stream1_expanded.splitfrac*feed.flow[b] :   0.0 :   True
                  c :   0.0 : stream1_expanded.flow[c] - stream1_expanded.splitfrac*feed.flow[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                         : Upper : Active
                None :   0.0 : stream1_expanded.mass - stream1_expanded.splitfrac*feed.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                 : Upper : Active
                None :   0.0 : feed.temp - tru.temp :   0.0 :   True

        6 Declarations: flow splitfrac flow_split mass mass_split temp_equality
    stream2_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                               : Upper : Active
                  a :   0.0 : stream2_expanded.flow[a] - stream2_expanded.splitfrac*feed.flow[a] :   0.0 :   True
                  b :   0.0 : stream2_expanded.flow[b] - stream2_expanded.splitfrac*feed.flow[b] :   0.0 :   True
                  c :   0.0 : stream2_expanded.flow[c] - stream2_expanded.splitfrac*feed.flow[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                         : Upper : Active
                None :   0.0 : stream2_expanded.mass - stream2_expanded.splitfrac*feed.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                  : Upper : Active
                None :   0.0 : feed.temp - prod.temp :   0.0 :   True

        6 Declarations: flow splitfrac flow_split mass mass_split temp_equality
    stream3_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True :  Reals
                  b :     0 :  None :  None : False :  True :  Reals
                  c :     0 :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                               : Upper : Active
                  a :   0.0 : stream3_expanded.flow[a] - stream3_expanded.splitfrac*feed.flow[a] :   0.0 :   True
                  b :   0.0 : stream3_expanded.flow[b] - stream3_expanded.splitfrac*feed.flow[b] :   0.0 :   True
                  c :   0.0 : stream3_expanded.flow[c] - stream3_expanded.splitfrac*feed.flow[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                         : Upper : Active
                None :   0.0 : stream3_expanded.mass - stream3_expanded.splitfrac*feed.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                   : Upper : Active
                None :   0.0 : feed.temp - node1.temp :   0.0 :   True

        6 Declarations: flow splitfrac flow_split mass mass_split temp_equality
    stream4_expanded : Size=1, Index=None, Active=True
        3 Constraint Declarations
            flow_equality : Size=3, Index=comp, Active=True
                Key : Lower : Body                          : Upper : Active
                  a :   0.0 : node1.flow[a] - node2.flow[a] :   0.0 :   True
                  b :   0.0 : node1.flow[b] - node2.flow[b] :   0.0 :   True
                  c :   0.0 : node1.flow[c] - node2.flow[c] :   0.0 :   True
            mass_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                    : Upper : Active
                None :   0.0 : node1.mass - node2.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                    : Upper : Active
                None :   0.0 : node1.temp - node2.temp :   0.0 :   True

        3 Declarations: flow_equality mass_equality temp_equality
    stream5_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                  : Upper : Active
                  a :   0.0 : stream5_expanded.flow[a] - stream5_expanded.splitfrac*tru.flow_out[a] :   0.0 :   True
                  b :   0.0 : stream5_expanded.flow[b] - stream5_expanded.splitfrac*tru.flow_out[b] :   0.0 :   True
                  c :   0.0 : stream5_expanded.flow[c] - stream5_expanded.splitfrac*tru.flow_out[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                        : Upper : Active
                None :   0.0 : stream5_expanded.mass - stream5_expanded.splitfrac*tru.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                 : Upper : Active
                None :   0.0 : tru.temp - prod.temp :   0.0 :   True

        6 Declarations: flow mass temp_equality splitfrac flow_split mass_split
    stream6_expanded : Size=1, Index=None, Active=True
        2 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True :  Reals
                  b :     0 :  None :  None : False :  True :  Reals
                  c :     0 :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        1 Constraint Declarations
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                   : Upper : Active
                None :   0.0 : node2.temp - prod.temp :   0.0 :   True

        3 Declarations: flow mass temp_equality
    stream7_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                               : Upper : Active
                  a :   0.0 : stream7_expanded.flow[a] - stream7_expanded.splitfrac*feed.flow[a] :   0.0 :   True
                  b :   0.0 : stream7_expanded.flow[b] - stream7_expanded.splitfrac*feed.flow[b] :   0.0 :   True
                  c :   0.0 : stream7_expanded.flow[c] - stream7_expanded.splitfrac*feed.flow[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                         : Upper : Active
                None :   0.0 : stream7_expanded.mass - stream7_expanded.splitfrac*feed.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                   : Upper : Active
                None :   0.0 : feed.temp - multi.temp :   0.0 :   True

        6 Declarations: flow splitfrac flow_split mass mass_split temp_equality
    stream8_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                  : Upper : Active
                  a :   0.0 : stream8_expanded.flow[a] - stream8_expanded.splitfrac*tru.flow_out[a] :   0.0 :   True
                  b :   0.0 : stream8_expanded.flow[b] - stream8_expanded.splitfrac*tru.flow_out[b] :   0.0 :   True
                  c :   0.0 : stream8_expanded.flow[c] - stream8_expanded.splitfrac*tru.flow_out[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                        : Upper : Active
                None :   0.0 : stream8_expanded.mass - stream8_expanded.splitfrac*tru.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                  : Upper : Active
                None :   0.0 : tru.temp - multi.temp :   0.0 :   True

        6 Declarations: flow splitfrac flow_split mass mass_split temp_equality
    stream9_expanded : Size=1, Index=None, Active=True
        3 Var Declarations
            flow : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :  None :  None :  None : False :  True :  Reals
                  b :  None :  None :  None : False :  True :  Reals
                  c :  None :  None :  None : False :  True :  Reals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            splitfrac : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        3 Constraint Declarations
            flow_split : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                : Upper : Active
                  a :   0.0 : stream9_expanded.flow[a] - stream9_expanded.splitfrac*multi.flow[a] :   0.0 :   True
                  b :   0.0 : stream9_expanded.flow[b] - stream9_expanded.splitfrac*multi.flow[b] :   0.0 :   True
                  c :   0.0 : stream9_expanded.flow[c] - stream9_expanded.splitfrac*multi.flow[c] :   0.0 :   True
            mass_split : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                          : Upper : Active
                None :   0.0 : stream9_expanded.mass - stream9_expanded.splitfrac*multi.mass :   0.0 :   True
            temp_equality : Size=1, Index=None, Active=True
                Key  : Lower : Body                   : Upper : Active
                None :   0.0 : multi.temp - prod.temp :   0.0 :   True

        6 Declarations: flow mass temp_equality splitfrac flow_split mass_split
    tru : Size=1, Index=None, Active=True
        4 Var Declarations
            flow_in : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            flow_out : Size=3, Index=comp
                Key : Lower : Value : Upper : Fixed : Stale : Domain
                  a :     0 :  None :  None : False :  True : NonNegativeReals
                  b :     0 :  None :  None : False :  True : NonNegativeReals
                  c :     0 :  None :  None : False :  True : NonNegativeReals
            mass : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals
            temp : Size=1, Index=None
                Key  : Lower : Value : Upper : Fixed : Stale : Domain
                None :  None :  None :  None : False :  True :  Reals

        4 Constraint Declarations
            inlet_flow_insum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                  : Upper : Active
                  a :   0.0 : stream1_expanded.flow[a] + stream10_expanded.flow[a] - tru.flow_in[a] :   0.0 :   True
                  b :   0.0 : stream1_expanded.flow[b] + stream10_expanded.flow[b] - tru.flow_in[b] :   0.0 :   True
                  c :   0.0 : stream1_expanded.flow[c] + stream10_expanded.flow[c] - tru.flow_in[c] :   0.0 :   True
            inlet_mass_insum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                      : Upper : Active
                None :   0.0 : stream1_expanded.mass + stream10_expanded.mass - tru.mass :   0.0 :   True
            outlet_flow_outsum : Size=3, Index=comp, Active=True
                Key : Lower : Body                                                                  : Upper : Active
                  a :   0.0 : stream5_expanded.flow[a] + stream8_expanded.flow[a] - tru.flow_out[a] :   0.0 :   True
                  b :   0.0 : stream5_expanded.flow[b] + stream8_expanded.flow[b] - tru.flow_out[b] :   0.0 :   True
                  c :   0.0 : stream5_expanded.flow[c] + stream8_expanded.flow[c] - tru.flow_out[c] :   0.0 :   True
            outlet_mass_outsum : Size=1, Index=None, Active=True
                Key  : Lower : Body                                                     : Upper : Active
                None :   0.0 : stream5_expanded.mass + stream8_expanded.mass - tru.mass :   0.0 :   True

        2 Port Declarations
            inlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : tru.flow_in
                     : mass :    1 :    tru.mass
                     : temp :    1 :    tru.temp
            outlet : Size=1, Index=None
                Key  : Name : Size : Variable
                None : flow :    3 : tru.flow_out
                     : mass :    1 :     tru.mass
                     : temp :    1 :     tru.temp

        10 Declarations: flow_in flow_out mass temp inlet outlet inlet_flow_insum inlet_mass_insum outlet_flow_outsum outlet_mass_outsum

11 Arc Declarations
    stream0 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (tru.outlet, node1.port) :     True :  False
    stream1 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (feed.outlet, tru.inlet) :     True :  False
    stream10 : Size=1, Index=None, Active=False
        Key  : Ports                   : Directed : Active
        None : (multi.port, tru.inlet) :     True :  False
    stream2 : Size=1, Index=None, Active=False
        Key  : Ports                     : Directed : Active
        None : (feed.outlet, prod.inlet) :     True :  False
    stream3 : Size=1, Index=None, Active=False
        Key  : Ports                     : Directed : Active
        None : (feed.outlet, node1.port) :     True :  False
    stream4 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (node1.port, node2.port) :     True :  False
    stream5 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (tru.outlet, prod.inlet) :     True :  False
    stream6 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (node2.port, prod.inlet) :     True :  False
    stream7 : Size=1, Index=None, Active=False
        Key  : Ports                     : Directed : Active
        None : (feed.outlet, multi.port) :     True :  False
    stream8 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (tru.outlet, multi.port) :     True :  False
    stream9 : Size=1, Index=None, Active=False
        Key  : Ports                    : Directed : Active
        None : (multi.port, prod.inlet) :     True :  False

28 Declarations: comp feed tru node1 node2 multi prod stream0 stream1 stream2 stream3 stream4 stream5 stream6 stream7 stream8 stream9 stream10 stream1_expanded stream2_expanded stream3_expanded stream4_expanded stream5_expanded stream6_expanded stream7_expanded stream8_expanded stream9_expanded stream10_expanded
""")


if __name__ == "__main__":
    unittest.main()
