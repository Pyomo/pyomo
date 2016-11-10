#  _________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2014 Sandia Corporation.
#  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
#  the U.S. Government retains certain rights in this software.
#  This software is distributed under the BSD License.
#  _________________________________________________________________________
#
# Unit Tests for Elements of a Model
#
# TestSimpleVar                Class for testing single variables
# TestArrayVar                Class for testing array of variables
#

import os
from os.path import abspath, dirname
currdir = dirname(abspath(__file__))+os.sep

import pyutilib.th as unittest
from six import StringIO

from pyomo.environ import *

class TestConnector(unittest.TestCase):

    def test_default_scalar_constructor(self):
        model = ConcreteModel()
        model.c = Connector()
        self.assertEqual(len(model.c), 1)
        self.assertEqual(len(model.c.vars), 0)

        model = AbstractModel()
        model.c = Connector()
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
        model.c = Connector([1,2,3])
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.c[1].vars), 0)

        model = AbstractModel()
        model.c = Connector([1,2,3])
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
  
        pipe.OUT = Connector()
        pipe.OUT.add(pipe.flow, "flow")
        pipe.OUT.add(pipe.pOut, "pressure")
        self.assertEqual(len(pipe.OUT), 1)
        self.assertEqual(len(pipe.OUT.vars), 2)
        self.assertFalse(pipe.OUT.vars['flow'].is_expression())

        pipe.IN = Connector()
        pipe.IN.add(-pipe.flow, "flow")
        pipe.IN.add(pipe.pIn, "pressure")
        self.assertEqual(len(pipe.IN), 1)
        self.assertEqual(len(pipe.IN.vars), 2)
        self.assertTrue(pipe.IN.vars['flow'].is_expression())
        
    def test_add_indexed_vars(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Connector()
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

        pipe.OUT = Connector()
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


    def test_polynomial_degree(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Connector()
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

    def test_pprint(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Connector()
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
         :        flow :    1 : -1 * flow
         :    pressure :    1 : pIn
""")

        def _IN(m, i):
            return { 'pressure': pipe.pIn,
                     'flow': pipe.composition[i] * pipe.flow }

        pipe.IN = Connector(pipe.SPECIES, rule=_IN)
        os = StringIO()
        pipe.IN.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""IN : Size=3, Index=SPECIES
    Key : Name     : Size : Variable
      a :     flow :    1 : composition[a] * flow
        : pressure :    1 : pIn
      b :     flow :    1 : composition[b] * flow
        : pressure :    1 : pIn
      c :     flow :    1 : composition[c] * flow
        : pressure :    1 : pIn
""")
        
    def test_display(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var(initialize=10)
        pipe.composition = Var( pipe.SPECIES,
                                initialize=lambda m,i: ord(i)-ord('a') )
        pipe.pIn  = Var( within=NonNegativeReals, initialize=3.14 )

        pipe.OUT = Connector()
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
         :    pressure : 3.14
""")

        def _IN(m, i):
            return { 'pressure': pipe.pIn,
                     'flow': pipe.composition[i] * pipe.flow }

        pipe.IN = Connector(pipe.SPECIES, rule=_IN)
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

    def test_expand_single_scalar(self):
        m = ConcreteModel()
        m.x = Var()
        m.CON = Connector()
        m.CON.add(m.x, "x")

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == 1)
        m.nocon = Constraint(expr = m.x == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 3)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=1, Index=c.expanded_index, Active=True
    Key : Lower : Body : Upper : Active
      1 :   1.0 :    x :   1.0 :   True
""")


    def test_expand_scalar(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == 1)
        m.nocon = Constraint(expr = m.x == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 4)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=2, Index=c.expanded_index, Active=True
    Key : Lower : Body : Upper : Active
      1 :   1.0 :    x :   1.0 :   True
      2 :   1.0 :    y :   1.0 :   True
""")


    def test_expand_expression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.CON = Connector()
        m.CON.add(-m.x)
        m.CON.add(1 + m.y)

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == 1)
        m.nocon = Constraint(expr = m.x == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 4)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=2, Index=c.expanded_index, Active=True
    Key : Lower : Body   : Upper : Active
      1 :   1.0 : -1 * x :   1.0 :   True
      2 :   1.0 :  1 + y :   1.0 :   True
""")


    def test_expand_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2])
        m.y = Var()
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == 1)
        m.nocon = Constraint(expr = m.x[1] == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 5)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=3, Index=c.expanded_index, Active=True
    Key : Lower : Body : Upper : Active
      1 :   1.0 : x[1] :   1.0 :   True
      2 :   1.0 : x[2] :   1.0 :   True
      3 :   1.0 :    y :   1.0 :   True
""")


    def test_expand_empty_scalar(self):
        m = ConcreteModel()
        m.x = Var(bounds=(1,3))
        m.y = Var(domain=Binary)
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)
        m.ECON = Connector()

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON)
        m.nocon = Constraint(expr = m.x == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 4)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        self.assertIs( m.x.domain, m.component('ECON.auto.x').domain )
        self.assertIs( m.y.domain, m.component('ECON.auto.y').domain )
        self.assertEqual( m.x.bounds, m.component('ECON.auto.x').bounds )
        self.assertEqual( m.y.bounds, m.component('ECON.auto.y').bounds )

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=2, Index=c.expanded_index, Active=True
    Key : Lower : Body            : Upper : Active
      1 :   0.0 : x - ECON.auto.x :   0.0 :   True
      2 :   0.0 : y - ECON.auto.y :   0.0 :   True
""")


    def test_expand_empty_expression(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.CON = Connector()
        m.CON.add(-m.x, 'x')
        m.CON.add(1 + m.y, 'y')
        m.ECON = Connector()

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON)
        m.nocon = Constraint(expr = m.x == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 4)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=2, Index=c.expanded_index, Active=True
    Key : Lower : Body                 : Upper : Active
      1 :   0.0 : -1 * x - ECON.auto.x :   0.0 :   True
      2 :   0.0 :  1 + y - ECON.auto.y :   0.0 :   True
""")


    def test_expand_empty_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)
        m.ECON = Connector()

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON)
        m.nocon = Constraint(expr = m.x[1] == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)
        #m.pprint()

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 5)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)

        self.assertIs( m.x[1].domain, m.component('ECON.auto.x')[1].domain )
        self.assertIs( m.x[2].domain, m.component('ECON.auto.x')[2].domain )
        self.assertIs( m.y.domain, m.component('ECON.auto.y').domain )
        self.assertEqual( m.x[1].bounds, m.component('ECON.auto.x')[1].bounds )
        self.assertEqual( m.x[2].bounds, m.component('ECON.auto.x')[2].bounds )
        self.assertEqual( m.y.bounds, m.component('ECON.auto.y').bounds )

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=3, Index=c.expanded_index, Active=True
    Key : Lower : Body                  : Upper : Active
      1 :   0.0 : x[1] - ECON.auto.x[1] :   0.0 :   True
      2 :   0.0 : x[2] - ECON.auto.x[2] :   0.0 :   True
      3 :   0.0 :       y - ECON.auto.y :   0.0 :   True
""")

    def test_expand_multiple_empty_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)
        m.ECON2 = Connector()
        m.ECON1 = Connector()

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON1)
        m.d = Constraint(expr= m.ECON2 == m.ECON1)
        m.nocon = Constraint(expr = m.x[1] == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 3)

        TransformationFactory('core.expand_connectors').apply_to(m)
        #m.pprint()

        self.assertEqual(len(list(m.component_objects(Constraint))), 5)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 9)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)
        self.assertFalse(m.d.active)
        self.assertTrue(m.component('d.expanded').active)

        self.assertIs( m.x[1].domain, m.component('ECON1.auto.x')[1].domain )
        self.assertIs( m.x[2].domain, m.component('ECON1.auto.x')[2].domain )
        self.assertIs( m.y.domain, m.component('ECON1.auto.y').domain )
        self.assertEqual( m.x[1].bounds, m.component('ECON1.auto.x')[1].bounds )
        self.assertEqual( m.x[2].bounds, m.component('ECON1.auto.x')[2].bounds )
        self.assertEqual( m.y.bounds, m.component('ECON1.auto.y').bounds )

        self.assertIs( m.x[1].domain, m.component('ECON2.auto.x')[1].domain )
        self.assertIs( m.x[2].domain, m.component('ECON2.auto.x')[2].domain )
        self.assertIs( m.y.domain, m.component('ECON2.auto.y').domain )
        self.assertEqual( m.x[1].bounds, m.component('ECON2.auto.x')[1].bounds )
        self.assertEqual( m.x[2].bounds, m.component('ECON2.auto.x')[2].bounds )
        self.assertEqual( m.y.bounds, m.component('ECON2.auto.y').bounds )

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=3, Index=c.expanded_index, Active=True
    Key : Lower : Body                   : Upper : Active
      1 :   0.0 : x[1] - ECON1.auto.x[1] :   0.0 :   True
      2 :   0.0 : x[2] - ECON1.auto.x[2] :   0.0 :   True
      3 :   0.0 :       y - ECON1.auto.y :   0.0 :   True
""")

        os = StringIO()
        m.component('d.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d.expanded : Size=3, Index=d.expanded_index, Active=True
    Key : Lower : Body                              : Upper : Active
      1 :   0.0 : ECON2.auto.x[1] - ECON1.auto.x[1] :   0.0 :   True
      2 :   0.0 : ECON2.auto.x[2] - ECON1.auto.x[2] :   0.0 :   True
      3 :   0.0 :       ECON2.auto.y - ECON1.auto.y :   0.0 :   True
""")


    def test_expand_multiple_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)
        m.a1 = Var([1,2])
        m.a2 = Var([1,2])
        m.b1 = Var()
        m.b2 = Var()
        m.ECON2 = Connector()
        m.ECON2.add(m.a1,'x')
        m.ECON2.add(m.b1,'y')
        m.ECON1 = Connector()
        m.ECON1.add(m.a2,'x')
        m.ECON1.add(m.b2,'y')

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON1)
        m.d = Constraint(expr= m.ECON2 == m.ECON1)
        m.nocon = Constraint(expr = m.x[1] == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 3)

        TransformationFactory('core.expand_connectors').apply_to(m)
        #m.pprint()

        self.assertEqual(len(list(m.component_objects(Constraint))), 5)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 9)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)
        self.assertFalse(m.d.active)
        self.assertTrue(m.component('d.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=3, Index=c.expanded_index, Active=True
    Key : Lower : Body         : Upper : Active
      1 :   0.0 : x[1] - a2[1] :   0.0 :   True
      2 :   0.0 : x[2] - a2[2] :   0.0 :   True
      3 :   0.0 :       y - b2 :   0.0 :   True
""")

        os = StringIO()
        m.component('d.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d.expanded : Size=3, Index=d.expanded_index, Active=True
    Key : Lower : Body          : Upper : Active
      1 :   0.0 : a1[1] - a2[1] :   0.0 :   True
      2 :   0.0 : a1[2] - a2[2] :   0.0 :   True
      3 :   0.0 :       b1 - b2 :   0.0 :   True
""")


    def test_expand_implicit_indexed(self):
        m = ConcreteModel()
        m.x = Var([1,2], domain=Binary)
        m.y = Var(bounds=(1,3))
        m.CON = Connector()
        m.CON.add(m.x)
        m.CON.add(m.y)
        m.a2 = Var([1,2])
        m.b1 = Var()
        m.ECON2 = Connector(implicit=['x'])
        m.ECON2.add(m.b1,'y')
        m.ECON1 = Connector(implicit=['y'])
        m.ECON1.add(m.a2,'x')

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON1)
        m.d = Constraint(expr= m.ECON2 == m.CON)
        m.nocon = Constraint(expr = m.x[1] == 2)

        self.assertEqual(len(list(m.component_objects(Constraint))), 3)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 3)

        os = StringIO()
        m.ECON1.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""ECON1 : Size=1, Index=None
    Key  : Name : Size : Variable
    None :    x :    2 :       a2
         :    y :    - :     None
""")

        TransformationFactory('core.expand_connectors').apply_to(m)
        #m.pprint()

        os = StringIO()
        m.ECON1.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""ECON1 : Size=1, Index=None
    Key  : Name : Size : Variable
    None :    x :    2 :           a2
         :    y :    1 : ECON1.auto.y
""")

        self.assertEqual(len(list(m.component_objects(Constraint))), 5)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 9)
        self.assertTrue(m.nocon.active)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)
        self.assertFalse(m.d.active)
        self.assertTrue(m.component('d.expanded').active)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=3, Index=c.expanded_index, Active=True
    Key : Lower : Body             : Upper : Active
      1 :   0.0 :     x[1] - a2[1] :   0.0 :   True
      2 :   0.0 :     x[2] - a2[2] :   0.0 :   True
      3 :   0.0 : y - ECON1.auto.y :   0.0 :   True
""")

        os = StringIO()
        m.component('d.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d.expanded : Size=3, Index=d.expanded_index, Active=True
    Key : Lower : Body                   : Upper : Active
      1 :   0.0 : ECON2.auto.x[1] - x[1] :   0.0 :   True
      2 :   0.0 : ECON2.auto.x[2] - x[2] :   0.0 :   True
      3 :   0.0 :                 b1 - y :   0.0 :   True
""")


    def test_varlist_aggregator(self):
        m = ConcreteModel()
        m.flow = VarList()
        m.phase = Var(bounds=(1,3))
        m.CON = Connector()
        m.CON.add( m.flow,
                   aggregate=lambda m,v: sum(v[i] for i in v) == 0 )
        m.CON.add(m.phase)
        m.ECON2 = Connector()
        m.ECON1 = Connector()

        # 2 constraints: one has a connector, the other doesn't.  The
        # former should be deactivated and expanded, the latter should
        # be left untouched.
        m.c = Constraint(expr= m.CON == m.ECON1)
        m.d = Constraint(expr= m.ECON2 == m.CON)

        self.assertEqual(len(list(m.component_objects(Constraint))), 2)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 2)

        TransformationFactory('core.expand_connectors').apply_to(m)
        #m.pprint()

        self.assertEqual(len(list(m.component_objects(Constraint))), 5)
        self.assertEqual(len(list(m.component_data_objects(Constraint))), 7)
        self.assertFalse(m.c.active)
        self.assertTrue(m.component('c.expanded').active)
        self.assertFalse(m.d.active)
        self.assertTrue(m.component('d.expanded').active)

        self.assertEqual( len(m.flow), 2 )

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=2, Index=c.expanded_index, Active=True
    Key : Lower : Body                      : Upper : Active
      1 :   0.0 : flow[1] - ECON1.auto.flow :   0.0 :   True
      2 :   0.0 :  phase - ECON1.auto.phase :   0.0 :   True
""")

        os = StringIO()
        m.component('d.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""d.expanded : Size=2, Index=d.expanded_index, Active=True
    Key : Lower : Body                      : Upper : Active
      1 :   0.0 : ECON2.auto.flow - flow[2] :   0.0 :   True
      2 :   0.0 :  ECON2.auto.phase - phase :   0.0 :   True
""")

        os = StringIO()
        m.component('CON.flow.aggregate').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""CON.flow.aggregate : Size=1, Index=None, Active=True
    Key  : Lower : Body              : Upper : Active
    None :   0.0 : flow[1] + flow[2] :   0.0 :   True
""")

        os = StringIO()
        m.CON.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""CON : Size=1, Index=None
    Key  : Name  : Size : Variable
    None :  flow :    * :     flow
         : phase :    1 :    phase
""")


    def test_indexed_connector(self):
        m = ConcreteModel()
        m.x = Var(initialize=1, domain=Reals)
        m.y = Var(initialize=2, domain=Reals)
        m.c = Connector([1,2])
        m.c[1].add(m.x, name='v')
        m.c[2].add(m.y, name='v')

        m.eq = Constraint(expr=m.c[1] == m.c[2])

        TransformationFactory('core.expand_connectors').apply_to(m)

        os = StringIO()
        m.component('eq.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""eq.expanded : Size=1, Index=eq.expanded_index, Active=True
    Key : Lower : Body  : Upper : Active
      1 :   0.0 : x - y :   0.0 :   True
""")


if __name__ == "__main__":
    unittest.main()
