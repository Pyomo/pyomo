import pyutilib.th as unittest
import pyomo.environ as pe
import pyomo.gdp as gdp
from pyomo.gdp.util import check_model_algebraic
from pyomo.common.log import LoggingIntercept
import logging
from six import StringIO


class TestGDPReclassificationError(unittest.TestCase):
    def test_disjunct_not_in_disjunction(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.d1 = gdp.Disjunct()
        m.d1.c = pe.Constraint(expr=m.x == 1)
        m.d2 = gdp.Disjunct()
        m.d2.c = pe.Constraint(expr=m.x == 0)
        pe.TransformationFactory('gdp.bigm').apply_to(m)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.WARNING):
            check_model_algebraic(m)
        self.assertRegexpMatches( log.getvalue(), 
                                  '.*not found in any Disjunctions.*')

    def test_disjunct_not_in_active_disjunction(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.d1 = gdp.Disjunct()
        m.d1.c = pe.Constraint(expr=m.x == 1)
        m.d2 = gdp.Disjunct()
        m.d2.c = pe.Constraint(expr=m.x == 0)
        m.disjunction = gdp.Disjunction(expr=[m.d1, m.d2])
        m.disjunction.deactivate()
        pe.TransformationFactory('gdp.bigm').apply_to(m)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.WARNING):
            check_model_algebraic(m)
        self.assertRegexpMatches(log.getvalue(), 
                                 '.*While it participates in a Disjunction, '
                                 'that Disjunction is currently deactivated.*')
