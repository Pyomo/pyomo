#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.util import check_model_algebraic
from pyomo.common.log import LoggingIntercept
import logging
from io import StringIO


class TestGDPReclassificationError(unittest.TestCase):
    def test_disjunct_not_in_disjunction(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.d1 = Disjunct()
        m.d1.c = pyo.Constraint(expr=m.x == 1)
        m.d2 = Disjunct()
        m.d2.c = pyo.Constraint(expr=m.x == 0)

        pyo.TransformationFactory('gdp.bigm').apply_to(m)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.WARNING):
            check_model_algebraic(m)
        self.assertRegex(log.getvalue(), '.*not found in any Disjunctions.*')

    def test_disjunct_not_in_active_disjunction(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.d1 = Disjunct()
        m.d1.c = pyo.Constraint(expr=m.x == 1)
        m.d2 = Disjunct()
        m.d2.c = pyo.Constraint(expr=m.x == 0)
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        m.disjunction.deactivate()
        pyo.TransformationFactory('gdp.bigm').apply_to(m)
        log = StringIO()
        with LoggingIntercept(log, 'pyomo.gdp', logging.WARNING):
            check_model_algebraic(m)
        self.assertRegex(
            log.getvalue(),
            '.*While it participates in a Disjunction, '
            'that Disjunction is currently deactivated.*',
        )
