import pyutilib.th as unittest
import pyomo.environ as pe
import pyomo.gdp as gdp


class TestGDPReclassificationError(unittest.TestCase):
    def test_disjunct_not_in_disjunction(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.d1 = gdp.Disjunct()
        m.d1.c = pe.Constraint(expr=m.x == 1)
        m.d2 = gdp.Disjunct()
        m.d2.c = pe.Constraint(expr=m.x == 0)
        with self.assertRaisesRegexp(
                gdp.GDP_Error, '.*not found in any Disjunctions.*'):
            pe.TransformationFactory('gdp.bigm').apply_to(m)

    def test_disjunct_not_in_active_disjunction(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.d1 = gdp.Disjunct()
        m.d1.c = pe.Constraint(expr=m.x == 1)
        m.d2 = gdp.Disjunct()
        m.d2.c = pe.Constraint(expr=m.x == 0)
        m.disjunction = gdp.Disjunction(expr=[m.d1, m.d2])
        m.disjunction.deactivate()
        with self.assertRaisesRegexp(
                gdp.GDP_Error, '.*While it participates in a Disjunction, '
                'that Disjunction is currently deactivated.*'):
            pe.TransformationFactory('gdp.bigm').apply_to(m)
