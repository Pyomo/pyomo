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

    def test_pprint(self):
        pipe = ConcreteModel()
        pipe.SPECIES = Set(initialize=['a','b','c'])
        pipe.flow = Var()
        pipe.composition = Var(pipe.SPECIES)
        pipe.pIn  = Var( within=NonNegativeReals )

        pipe.OUT = Connector()
        pipe.OUT.add(-pipe.flow, "flow")
        pipe.OUT.add(pipe.composition, "composition")
        pipe.OUT.add(pipe.pIn, "pressure")

        os = StringIO()
        pipe.OUT.pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""OUT : Size=1, Index=None
    Key  : Name        : Size : Variable
    None :    pressure :    1 : pIn
         :        flow :    1 : -1 * flow
         : composition :    3 : composition
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
      a : pressure :    1 : pIn
        :     flow :    1 : composition[a] * flow
      b : pressure :    1 : pIn
        :     flow :    1 : composition[b] * flow
      c : pressure :    1 : pIn
        :     flow :    1 : composition[c] * flow
""")
        
    def test_single_scalar_expand(self):
        m = ConcreteModel()
        m.x = Var()
        m.CON = Connector()
        m.CON.add(m.x, "x")

        m.c = Constraint(expr= m.CON == 1)

        TransformationFactory('core.expand_connectors').apply_to(m)

        os = StringIO()
        m.component('c.expanded').pprint(ostream=os)
        self.assertEqual(os.getvalue(),
"""c.expanded : Size=1, Index=c.expanded_index, Active=True
    Key : Lower : Body : Upper : Active
      1 :   1.0 :    x :   1.0 :   True
""")


if __name__ == "__main__":
    unittest.main()
