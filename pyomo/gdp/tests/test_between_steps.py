#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.common.unittest as unittest
from pyomo.environ import (TransformationFactory, Constraint, ConcreteModel,
                           Var, RangeSet, Objective, maximize)
from pyomo.gdp import Disjunct, Disjunction
import pyomo.gdp.tests.common_tests as ct

from nose.tools import set_trace

class CommonTests:
    def diff_apply_to_and_create_using(self, model):
        ct.diff_apply_to_and_create_using(self, model, 'gdp.between_steps')

class PaperTwoCircleExample(unittest.TestCase, CommonTests):
    def makeModel(self):
        m = ConcreteModel()
        m.I = RangeSet(1,4)
        m.x = Var(m.I, bounds=(-2,6))

        m.disjunction = Disjunction(expr=[[sum(m.x[i]**2 for i in m.I) <= 1],
                                          [sum((3 - m.x[i])**2 for i in m.I) <=
                                           1]])

        m.obj = Objective(expr=m.x[2] - m.x[1], sense=maximize)

        return m

    def test_something(self):
        m = self.makeModel()

        TransformationFactory('gdp.between_steps').apply_to(
            m,
            variable_partitions=[[m.x[1], m.x[2]], [m.x[3], m.x[4]]])

        set_trace()
        # This is right!
