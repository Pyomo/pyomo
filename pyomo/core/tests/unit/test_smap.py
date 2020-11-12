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
# Unit Tests for SymbolMap
#

import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, Set, Var, Objective, Constraint, Block, SymbolMap, TextLabeler
from pyomo.core.base.symbol_map import symbol_map_from_instance


class Test(unittest.TestCase):

    def setUp(self):
        #
        # Create model instance
        #
        model = ConcreteModel()
        model.A = Set(initialize=[1,2,3])
        model.x = Var()
        model.y = Var(model.A, dense=True)
        model.o1 = Objective(expr=model.x)
        def o2_(model, i):
            return model.x
        model.o2 = Objective(model.A, rule=o2_)
        model.c1 = Constraint(expr=model.x >= 1)
        def c2_(model, i):
            if i == 1:
                return model.x <= 2
            elif i == 2:
                return (3, model.x, 4)
            else:
                return model.x == 5
        model.c2 = Constraint(model.A, rule=c2_)
        model.b = Block()
        model.b.x = Var()
        model.b.y = Var(model.A, dense=True)
        self.instance = model

    def tearDown(self):
        self.instance = None

    def test_add(self):
        smap = SymbolMap()        
        smap.addSymbol(self.instance.x, "x")
        smap.addSymbol(self.instance.y[1], "y[1]")
        self.assertEqual( set(smap.bySymbol.keys()), set(['x','y[1]']))

    def test_adds(self):
        smap = SymbolMap()        
        labeler = TextLabeler()
        smap.addSymbols((obj,labeler(obj)) for obj in self.instance.component_data_objects(Var))
        self.assertEqual( set(smap.bySymbol.keys()), set(['x','y(1)','y(2)','y(3)','b_x','b_y(1)','b_y(2)','b_y(3)']))

    def test_create(self):
        smap = SymbolMap()        
        labeler = TextLabeler()
        smap.createSymbol(self.instance.x, labeler)
        smap.createSymbol(self.instance.y[1], labeler)
        self.assertEqual( set(smap.bySymbol.keys()), set(['x','y(1)']))

    def test_creates(self):
        smap = SymbolMap()        
        labeler = TextLabeler()
        smap.createSymbols(self.instance.component_data_objects(Var), labeler)
        self.assertEqual( set(smap.bySymbol.keys()), set(['x','y(1)','y(2)','y(3)','b_x','b_y(1)','b_y(2)','b_y(3)']))

    def test_get(self):
        smap = SymbolMap()        
        labeler = TextLabeler()
        self.assertEqual('x', smap.getSymbol(self.instance.x, labeler))
        self.assertEqual('y(1)', smap.getSymbol(self.instance.y[1], labeler))
        self.assertEqual( set(smap.bySymbol.keys()), set(['x','y(1)']))
        self.assertEqual('x', smap.getSymbol(self.instance.x, labeler))

    def test_alias_and_getObject(self):
        smap = SymbolMap()        
        smap.addSymbol(self.instance.x, 'x')
        smap.alias(self.instance.x, 'X')
        self.assertEqual( set(smap.bySymbol.keys()), set(['x']))
        self.assertEqual( set(smap.aliases.keys()), set(['X']))
        self.assertEqual( id(smap.getObject('x')), id(self.instance.x) )
        self.assertEqual( id(smap.getObject('X')), id(self.instance.x) )

    def test_from_instance(self):
        smap = symbol_map_from_instance(self.instance)
        self.assertEqual( set(smap.bySymbol.keys()), set(['x','y(1)','y(2)','y(3)','b_x','b_y(1)','b_y(2)','b_y(3)','o1','o2(1)','o2(2)','o2(3)','c1','c2(1)','c2(2)','c2(3)']))
        self.assertEqual( set(smap.aliases.keys()), set(['c_e_c2(3)_',
                '__default_objective__',
                'c_u_c2(1)_',
                'c_l_c1_',
                'r_u_c2(2)_',
                'r_l_c2(2)_']) )

    def test_error1(self):
        smap = SymbolMap()        
        labeler = TextLabeler()
        self.assertEqual('x', smap.getSymbol(self.instance.x, labeler))
        class FOO(object):
            def __call__(self, *args):
                return 'x'
        labeler = FOO()
        try:
            self.assertEqual('x', smap.getSymbol(self.instance.y[1], labeler))
            self.fail("Expected RuntimeError")
        except RuntimeError:
            pass

    def test_error2(self):
        smap = SymbolMap()        
        smap.addSymbol(self.instance.x, 'x')
        smap.alias(self.instance.x, 'X')
        self.assertEqual( id(smap.getObject('x')), id(self.instance.x) )
        self.assertEqual( id(smap.getObject('X')), id(self.instance.x) )
        self.assertEqual( id(smap.getObject('y')), id(SymbolMap.UnknownSymbol) )

if __name__ == "__main__":
    unittest.main()
