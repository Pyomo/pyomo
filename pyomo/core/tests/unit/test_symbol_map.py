import pickle

import pyutilib.th as unittest
import pyomo.environ
from pyomo.core.expr.symbol_map import SymbolMap
from pyomo.core.kernel.variable import variable

class TestSymbolMap(unittest.TestCase):

    def test_no_labeler(self):
        s = SymbolMap()
        v = variable()
        self.assertEquals(str(v), s.getSymbol(v))

    def test_existing_alias(self):
        s = SymbolMap()
        v1 = variable()
        s.alias(v1, "v")
        self.assertIs(s.aliases["v"](), v1)
        v2 = variable()
        with self.assertRaises(RuntimeError):
            s.alias(v2, "v")
        s.alias(v1, "A")
        self.assertIs(s.aliases["v"](), v1)
        self.assertIs(s.aliases["A"](), v1)
        s.alias(v1, "A")
        self.assertIs(s.aliases["v"](), v1)
        self.assertIs(s.aliases["A"](), v1)

if __name__ == "__main__":
    unittest.main()
