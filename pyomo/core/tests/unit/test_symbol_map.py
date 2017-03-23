import pickle

import pyutilib.th as unittest
import pyomo.environ
from pyomo.core.kernel.symbol_map import SymbolMap
from pyomo.core.kernel.component_variable import variable

class TestSymbolMap(unittest.TestCase):

    def test_no_pickle(self):
        s = SymbolMap()
        with self.assertRaises(RuntimeError):
            s.__getstate__()
        with self.assertRaises(RuntimeError):
            s.__setstate__(None)
        s.addSymbol(variable(), "x")
        with self.assertRaises(RuntimeError):
            s.__getstate__()
        with self.assertRaises(RuntimeError):
            s.__setstate__(None)

    def test_no_labeler(self):
        s = SymbolMap()
        v = variable()
        with self.assertRaises(RuntimeError):
            s.getSymbol(v)

    def test_existing_alias(self):
        s = SymbolMap()
        v1 = variable()
        s.alias(v1, "v")
        v2 = variable()
        with self.assertRaises(RuntimeError):
            s.alias(v2, "v")
        s.alias(v1, "A")

if __name__ == "__main__":
    unittest.main()
