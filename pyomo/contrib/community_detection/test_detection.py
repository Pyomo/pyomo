#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
from __future__ import division

import logging
from math import pi

from six import StringIO

import pyutilib.th as unittest
from pyomo.contrib.community_detection.detection import detect_constraint_communities, detect_variable_communities
from pyomo.common.log import LoggingIntercept
from pyomo.core import (
    ConcreteModel, Expression, Var, acos, asin, atan, cos, exp, quicksum, sin,
    tan, value,
    ComponentMap, log)
from pyomo.core.expr.current import identify_variables


class TestMcCormick(unittest.TestCase):

    def test_outofbounds(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 5), initialize=2)
        result = detect_variable_communities(m)
        self.assertEqual(result, dict())







if __name__ == '__main__':
    unittest.main()
