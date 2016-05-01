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

from pyomo.environ import *

class TestConnector(unittest.TestCase):

    def test_default_scalar_constructor(self):
        model = ConcreteModel()
        model.c = Connector()
        self.assertEqual(len(model.c), 1)
        self.assertEqual(len(model.c.vars), 0)

        model = AbstractModel()
        model.c = Connector()
        self.assertEqual(len(model.c), 1)
        self.assertEqual(len(model.c.vars), 0)

    def test_default_indexed_constructor(self):
        model = ConcreteModel()
        model.c = Connector([1,2,3])
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.c[1].vars), 0)

        model = AbstractModel()
        model.c = Connector([1,2,3])
        self.assertEqual(len(model.c), 3)
        self.assertEqual(len(model.c[1].vars), 0)

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
        pipe.OUT.add(pipe.flow, "flow")
        pipe.OUT.add(pipe.composition, "composition")
        pipe.OUT.add(pipe.pIn, "pressure")

        pipe.pprint()
        
    def test_single_scalar_expand(self):
        m = ConcreteModel()
        m.x = Var()
        m.CON = Connector()
        m.CON.add(m.x, "x")

        m.c = Constraint(expr= m.CON == 1)

        TransformationFactory('core.expand_connectors').apply_to(m)

        m.pprint()

if __name__ == "__main__":
    unittest.main()
