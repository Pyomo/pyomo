import pyomo.environ as pe
from pyomo.util.report_scaling import report_scaling
import logging
from pyomo.common import unittest
from pyomo.common.log import LoggingIntercept
from io import StringIO
import re


class TestReportScaling(unittest.TestCase):
    def test_report_scaling(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(list(range(5)))
        m.c = pe.Constraint(list(range(3)))

        m.x[0].setlb(-1)
        m.x[0].setub(1)
        m.x[1].setlb(1e5)
        m.x[1].setub(1e7)
        m.x[2].setlb(-1e5)
        m.x[2].setub(0)
        m.x[3].setlb(-20)
        m.x[3].setub(20)

        m.obj1 = pe.Objective(expr=1e-8 * m.x[0] + pe.exp(m.x[3]) + m.x[1] * m.x[2])
        m.obj2 = pe.Objective(expr=m.x[0] * m.x[3] + m.x[1] ** 2)

        m.c[0] = m.x[0] + m.x[3] == 0
        m.c[1] = 1 / m.x[1] == 1
        m.c[2] = m.x[1] * m.x[3] == 1

        out = StringIO()
        with LoggingIntercept(out, 'pyomo.util.report_scaling', level=logging.INFO):
            report_scaling(m)

        s = out.getvalue()
        match = re.search('\n\nThe following variables are not bounded.*\s*Var\s*LB\s*UB\s*x\[4\]\s*-inf\s*inf\s*The '
                          'following variables have large bounds.*\s*Var\s*LB\s*UB\s*x\[1\]\s*.*\s*x\[2\]\s*.*\s*.*\s*'
                          'The following objectives have potentially large coefficients.*\s*obj1\s*Var\s*Coef LB\s*Coef '
                          'UB\s*x\[3\]\s*.*\s*.*\s*x\[1\]\s*.*\s*.*\s*x\[2\]\s*.*\s*.*\s*obj2\s*Var\s*Coef LB\s*Coef '
                          'UB\s*x\[1\]\s*.*\s*.*\s*The following objectives have small coefficients.*\s*obj1\s*Var\s*'
                          'Coef LB\s*Coef UB\s*x\[0\]\s*.*\s*.*\s*The following constraints have potentially large '
                          'coefficients.*\s*c\[2\]\s*Var\s*Coef LB\s*Coef UB\s*x\[3\]\s*.*\s*.*\s*The following '
                          'constraints have small coefficients.*\s*c\[1\]\s*Var\s*Coef LB\s*Coef UB\s*x\[1\]\s*.*'
                          '\s*.*\s*The following constraints have bodies with large bounds.*\s*Constraint\s*LB\s*UB\s*'
                          'c\[2\]\s*.*\s*.*\s*', s)
        self.assertEqual(match.span(), (0, len(s)))
