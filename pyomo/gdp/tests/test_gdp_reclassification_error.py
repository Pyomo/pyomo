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
        try:
            pe.TransformationFactory('gdp.bigm').apply_to(m)
        except gdp.GDP_Error as err:
            s = str(err)
            s = s.replace('\n', '')
            s = s.replace(' ', '')
            self.assertTrue('notfoundonanyDisjunctions' in s)

    def test_disjunct_not_in_active_disjunction(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.d1 = gdp.Disjunct()
        m.d1.c = pe.Constraint(expr=m.x == 1)
        m.d2 = gdp.Disjunct()
        m.d2.c = pe.Constraint(expr=m.x == 0)
        m.disjunction = gdp.Disjunction(expr=[m.d1, m.d2])
        m.disjunction.deactivate()
        try:
            pe.TransformationFactory('gdp.bigm').apply_to(m)
        except gdp.GDP_Error as err:
            s = str(err)
            s = s.replace('\n', '')
            s = s.replace(' ', '')
            self.assertTrue('notfoundonanyactiveDisjunctions' in s)
