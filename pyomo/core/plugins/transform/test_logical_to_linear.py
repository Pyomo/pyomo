import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, AbstractModel, BooleanVar, LogicalStatement, TransformationFactory, RangeSet


class TestLogicalToLinearTransformation(unittest.TestCase):
    def test_single_statement(self):
        m = ConcreteModel()
        m.x = BooleanVar()
        m.y = BooleanVar()

        m.p = LogicalStatement(expr=m.x.implies(m.y))

        TransformationFactory('core.logical_to_linear').apply_to(m)

        m.pprint()

    def test_longer_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)

        m.p = LogicalStatement(expr=m.Y[1] >> (m.Y[2] | m.Y[3]))

        TransformationFactory('core.logical_to_linear').apply_to(m)

        m.pprint()


if __name__ == "__main__":
    unittest.main()
