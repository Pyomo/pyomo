import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, AbstractModel, BooleanVar, LogicalStatement, TransformationFactory


class TestLogicalToLinearTransformation(unittest.TestCase):
    def test_single_statement(self):
        m = ConcreteModel()
        m.x = BooleanVar()
        m.y = BooleanVar()

        m.p = LogicalStatement(expr=m.x.implies(m.y))

        TransformationFactory('core.logical_to_linear').apply_to(m)


if __name__ == "__main__":
    unittest.main()
