import pyutilib.th as unittest

from pyomo.core.expr.logical_expr import AtLeast, AtMost, Exactly
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import update_boolean_vars_from_binary
from pyomo.environ import ConcreteModel, BooleanVar, LogicalStatement, TransformationFactory, RangeSet, \
    Var, Constraint
from pyomo.gdp import Disjunct, Disjunction


def _generate_boolean_model(nvars):
    m = ConcreteModel()
    m.s = RangeSet(nvars)
    m.Y = BooleanVar(m.s)
    return m


@unittest.skipUnless(sympy_available, "Sympy not available")
class TestAtomicTransformations(unittest.TestCase):
    def test_implies(self):
        m = ConcreteModel()
        m.x = BooleanVar()
        m.y = BooleanVar()
        m.p = LogicalStatement(expr=m.x.implies(m.y))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

    def test_literal(self):
        m = ConcreteModel()
        m.Y = BooleanVar()
        m.p = LogicalStatement(expr=m.Y)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

    def test_constant_True(self):
        m = ConcreteModel()
        m.p = LogicalStatement(expr=True)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

    def test_nothing_to_do(self):
        m = ConcreteModel()
        m.p = LogicalStatement()
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()


@unittest.skipUnless(sympy_available, "Sympy not available")
class TestLogicalToLinearTransformation(unittest.TestCase):
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
        m = _generate_boolean_model(4)
        m.p = LogicalStatement(expr=AtLeast(1, AtLeast(2, m.Y[1], m.Y[1] | m.Y[2], m.Y[2]) | m.Y[3], m.Y[4]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()
        # TODO check if accurate

    def test_link_with_gdp_indicators(self):
        m = _generate_boolean_model(4)
        m.d1 = Disjunct()
        m.d2 = Disjunct()
        m.x = Var()
        m.dd = Disjunct([1, 2])
        m.d1.c = Constraint(expr=m.x >= 2)
        m.d2.c = Constraint(expr=m.x <= 10)
        m.dd[1].c = Constraint(expr=m.x >= 5)
        m.dd[2].c = Constraint(expr=m.x <= 6)
        m.Y[1].set_binary_var(m.d1.indicator_var)
        m.Y[2].set_binary_var(m.d2.indicator_var)
        m.Y[3].set_binary_var(m.dd[1].indicator_var)
        m.Y[4].set_binary_var(m.dd[2].indicator_var)
        m.p = LogicalStatement(expr=m.Y[1] >> m.Y[3] | m.Y[4])
        m.p2 = LogicalStatement(expr=AtMost(2, *m.Y[:]))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()

    def test_gdp_nesting(self):
        m = _generate_boolean_model(2)
        m.disj = Disjunction(expr=[
            [m.Y[1] >> m.Y[2]],
            [m.Y[2] == False]
        ])
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.pprint()


@unittest.skipUnless(sympy_available, "Sympy not available")
class TestLogicalToLinearBackmap(unittest.TestCase):
    def test_backmap(self):
        m = _generate_boolean_model(3)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 1
        m.Y_asbinary[2].value = 0
        update_boolean_vars_from_binary(m)
        self.assertTrue(m.Y[1].value)
        self.assertFalse(m.Y[2].value)
        self.assertIsNone(m.Y[3].value)

    def test_backmap_noninteger(self):
        m = _generate_boolean_model(2)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 0.9
        update_boolean_vars_from_binary(m, integer_tolerance=0.1)
        self.assertTrue(m.Y[1].value)
        with self.assertRaisesRegexp(ValueError, r"Binary variable has non-\{0,1\} value"):
            update_boolean_vars_from_binary(m)


if __name__ == "__main__":
    unittest.main()
