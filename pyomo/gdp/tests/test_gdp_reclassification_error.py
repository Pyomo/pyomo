#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyutilib.th as unittest
from pyomo.environ import ConcreteModel, Var, Constraint, TransformationFactory
from pyomo.gdp import Disjunct, Disjunction, GDP_Error


class TestGDPReclassificationError(unittest.TestCase):
    def test_disjunct_not_in_disjunction(self):
        m =  ConcreteModel()
        m.x =  Var()
        m.d1 =  Disjunct()
        m.d1.c =  Constraint(expr=m.x == 1)
        m.d2 =  Disjunct()
        m.d2.c =  Constraint(expr=m.x == 0)
        with self.assertRaisesRegexp(
                 GDP_Error, '.*not found in any Disjunctions.*'):
             TransformationFactory('gdp.bigm').apply_to(m)

    def test_disjunct_not_in_active_disjunction(self):
        m =  ConcreteModel()
        m.x =  Var()
        m.d1 =  Disjunct()
        m.d1.c =  Constraint(expr=m.x == 1)
        m.d2 =  Disjunct()
        m.d2.c =  Constraint(expr=m.x == 0)
        m.disjunction =  Disjunction(expr=[m.d1, m.d2])
        m.disjunction.deactivate()
        with self.assertRaisesRegexp(
                 GDP_Error, '.*While it participates in a Disjunction, '
                'that Disjunction is currently deactivated.*'):
             TransformationFactory('gdp.bigm').apply_to(m)
