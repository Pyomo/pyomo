import pyutilib.th as unittest

from pyomo.core.expr.logical_expr import AtLeast, AtMost, Exactly
from pyomo.environ import ConcreteModel, AbstractModel, BooleanVar, LogicalStatement, TransformationFactory, RangeSet, \
    Var


class TestLogicalToLinearTransformation(unittest.TestCase):
    def _generate_boolean_model(self, nvars):
        m = ConcreteModel()
        m.s = RangeSet(nvars)
        m.Y = BooleanVar(m.s)
        return m

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

        # TODO add in other statements
        # TODO add in asserts to make sure things are generated correctly

    def test_xfrm_special_atoms_nonroot(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalStatement(expr=m.Y[1] >> AtLeast(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        # m.pprint()

        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalStatement(expr=m.Y[1] >> AtMost(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        # m.pprint()

        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.p = LogicalStatement(expr=m.Y[1] >> Exactly(2, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        # m.pprint()

        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        m.x = Var(bounds=(1, 3))
        m.p = LogicalStatement(expr=m.Y[1] >> Exactly(m.x, m.Y[1], m.Y[2], m.Y[3]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

    def test_xfrm_atleast_nested(self):
        m = self._generate_boolean_model(4)
        m.p = LogicalStatement(expr=AtLeast(1, AtLeast(2, m.Y[1], m.Y[1] | m.Y[2], m.Y[2]) | m.Y[3], m.Y[4]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()
        # TODO check if accurate


if __name__ == "__main__":
    unittest.main()
