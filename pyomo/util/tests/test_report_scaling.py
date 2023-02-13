#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

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
        m.c = pe.Constraint(list(range(4)))
        m.p = pe.Param(initialize=0, mutable=True)

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
        m.c[3] = m.x[3] + m.p * m.x[0] == 1

        out = StringIO()
        with LoggingIntercept(out, 'pyomo.util.report_scaling', level=logging.INFO):
            report_scaling(m)

        expected = """

The following variables are not bounded. Please add bounds.
          LB          UB    Var
        -inf         inf    x[4]

The following variables have large bounds. Please scale them.
          LB          UB    Var
    1.00e+05    1.00e+07    x[1]
   -1.00e+05    0.00e+00    x[2]

The following objectives have potentially large coefficients. Please scale them.
obj1
         Coef LB     Coef UB    Var
        2.06e-09    4.85e+08    x[3]
       -1.00e+05    0.00e+00    x[1]
        1.00e+05    1.00e+07    x[2]

obj2
         Coef LB     Coef UB    Var
        2.00e+05    2.00e+07    x[1]

The following objectives have small coefficients.
obj1
         Coef LB     Coef UB    Var
        1.00e-08    1.00e-08    x[0]

The following constraints have potentially large coefficients. Please scale them.
c[2]
         Coef LB     Coef UB    Var
        1.00e+05    1.00e+07    x[3]

The following constraints have small coefficients.
c[1]
         Coef LB     Coef UB    Var
       -1.00e-10   -1.00e-14    x[1]

The following constraints have bodies with large bounds. Please scale them.
          LB          UB    Constraint
   -2.00e+08    2.00e+08    c[2]

"""

        self.assertEqual(out.getvalue(), expected)
