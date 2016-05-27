#!/usr/bin/env python
#
# RealProblem1.py

import sys
from pyomo.opt.blackbox import RealOptProblem

# @prob:
class RealProblem1(RealOptProblem):

    def __init__(self):
        RealOptProblem.__init__(self)
        self.lower=[0.0, -1.0, 1.0, None]
        self.upper=[None, 0.0, 2.0, -1.0]
        self.nvars=4

    def function_value(self, point):
        self.validate(point)
        return point.vars[0] - point.vars[1] + (point.vars[2]-1.5)**2 + (point.vars[3]+2)**4
# @:prob

# @main:
problem = RealProblem1()
problem.main(sys.argv)
# @:main
