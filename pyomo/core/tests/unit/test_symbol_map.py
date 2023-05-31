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

import pyomo.common.unittest as unittest
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable
from pyomo.environ import ConcreteModel, Var


class TestSymbolMap(unittest.TestCase):
    def test_no_labeler(self):
        s = SymbolMap()
        v = variable()
        self.assertEqual(str(v), s.getSymbol(v))

        s = SymbolMap()
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual('x', s.createSymbol(m.x))

        s = SymbolMap()
        m.y = Var([1, 2, 3])
        s.createSymbols(m.y.values())
        self.assertEqual(s.bySymbol, {'y[1]': m.y[1], 'y[2]': m.y[2], 'y[3]': m.y[3]})
        self.assertEqual(
            s.byObject, {id(m.y[1]): 'y[1]', id(m.y[2]): 'y[2]', id(m.y[3]): 'y[3]'}
        )

    def test_default_labeler(self):
        s = SymbolMap(lambda x: "_" + str(x))
        v = variable()
        self.assertEqual("_" + str(v), s.getSymbol(v))

        s = SymbolMap(lambda x: "_" + str(x))
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual('_x', s.createSymbol(m.x))

        s = SymbolMap(lambda x: "_" + str(x))
        m.y = Var([1, 2, 3])
        s.createSymbols(m.y.values())
        self.assertEqual(
            s.bySymbol, {'_y[1]': m.y[1], '_y[2]': m.y[2], '_y[3]': m.y[3]}
        )
        self.assertEqual(
            s.byObject, {id(m.y[1]): '_y[1]', id(m.y[2]): '_y[2]', id(m.y[3]): '_y[3]'}
        )

    def test_custom_labeler(self):
        labeler = lambda x, y: "^" + str(x) + y

        s = SymbolMap(lambda x: "_" + str(x))
        v = variable()
        self.assertEqual("^" + str(v) + "~", s.getSymbol(v, labeler, "~"))

        s = SymbolMap(lambda x: "_" + str(x))
        m = ConcreteModel()
        m.x = Var()
        self.assertEqual('^x~', s.createSymbol(m.x, labeler, "~"))

        s = SymbolMap(lambda x: "_" + str(x))
        m.y = Var([1, 2, 3])
        s.createSymbols(m.y.values(), labeler, "~")
        self.assertEqual(
            s.bySymbol, {'^y[1]~': m.y[1], '^y[2]~': m.y[2], '^y[3]~': m.y[3]}
        )
        self.assertEqual(
            s.byObject,
            {id(m.y[1]): '^y[1]~', id(m.y[2]): '^y[2]~', id(m.y[3]): '^y[3]~'},
        )

    def test_existing_alias(self):
        s = SymbolMap()
        v1 = variable()
        s.alias(v1, "v")
        self.assertIs(s.aliases["v"], v1)
        v2 = variable()
        with self.assertRaises(RuntimeError):
            s.alias(v2, "v")
        s.alias(v1, "A")
        self.assertIs(s.aliases["v"], v1)
        self.assertIs(s.aliases["A"], v1)
        s.alias(v1, "A")
        self.assertIs(s.aliases["v"], v1)
        self.assertIs(s.aliases["A"], v1)

    def test_add_symbol(self):
        s = SymbolMap()
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])
        s.addSymbol(m.x, 'x')
        self.assertEqual(s.bySymbol, {'x': m.x})
        self.assertEqual(s.byObject, {id(m.x): 'x'})
        with self.assertRaisesRegex(
            RuntimeError, r'SymbolMap.addSymbol\(\): duplicate symbol.'
        ):
            s.addSymbol(m.y, 'x')
        s = SymbolMap()
        s.addSymbol(m.x, 'x')
        with self.assertRaisesRegex(
            RuntimeError, r'SymbolMap.addSymbol\(\): duplicate object.'
        ):
            s.addSymbol(m.x, 'y')

    def test_add_symbols(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var([1, 2, 3])

        s = SymbolMap()
        s.addSymbols((m.y[i], str(i)) for i in (1, 2, 3))
        self.assertEqual(s.bySymbol, {'1': m.y[1], '2': m.y[2], '3': m.y[3]})
        self.assertEqual(
            s.byObject, {id(m.y[1]): '1', id(m.y[2]): '2', id(m.y[3]): '3'}
        )
        with self.assertRaisesRegex(
            RuntimeError, r'SymbolMap.addSymbols\(\): duplicate symbol.'
        ):
            s.addSymbols([(m.y, '1')])

        s = SymbolMap()
        s.addSymbols((m.y[i], str(i)) for i in (1, 2, 3))
        with self.assertRaisesRegex(
            RuntimeError, r'SymbolMap.addSymbols\(\): duplicate object.'
        ):
            s.addSymbols([(m.y[2], 'x')])


if __name__ == "__main__":
    unittest.main()
