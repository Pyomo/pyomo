"""Tests for the model size report utility."""
from pyomo.util.model_size import build_model_size_report
import pyutilib.th as unittest
from pyomo.core import ConcreteModel, Block ,Var, Constraint

from pyutilib.misc import import_file

from os.path import abspath, dirname, normpath, join
currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'gdp'))


class TestGDPopt(unittest.TestCase):
    """Tests for model size report utility."""

    def test_empty_model(self):
        """Test with an empty model."""
        empty_model = ConcreteModel()
        model_size = build_model_size_report(empty_model)
        for obj in model_size.active.values():
            self.assertEqual(obj, 0)

    @unittest.skip("Example file is not where it should be yet.")
    def test_eight_process(self):
        """Test with the eight process problem model."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        model_size = build_model_size_report(eight_process)
        self.assertEqual(model_size.active.variables, 30)

    def test_nested_blocks(self):
        """Test with nested blocks."""
        m = ConcreteModel()
        m.b = Block()
        m.inactive_b = Block()
        m.b.x = Var()
        m.b.x2 = Var()
        m.inactive_b.x = Var()
        m.b.c = Constraint(expr=m.b.x == m.b.x2)
        m.inactive_b.c = Constraint(expr=m.b.x == 1)
        m.inactive_b.c2 = Constraint(expr=m.inactive_b.x == 15)


if __name__ == '__main__':
    unittest.main()
