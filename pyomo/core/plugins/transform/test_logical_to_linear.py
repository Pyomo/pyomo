import pyutilib.th as unittest

from pyomo.core.expr.logical_expr import AtLeast
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

    def test_xfrm_atleast_statement(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalStatement(expr=AtLeast(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

        # TODO two below do not work yet

    def test_xfrm_atleast_nonroot(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalStatement(expr=m.Y[1] >> AtLeast(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

        # m = ConcreteModel()
        # m.s = RangeSet(3)
        # m.Y = BooleanVar(m.s)
        # m.p = LogicalStatement(expr=m.Y[1] >> AtLeast(2, m.Y))
        # TransformationFactory('core.logical_to_linear').apply_to(m)
        # m.pprint()


if __name__ == "__main__":
    unittest.main()
