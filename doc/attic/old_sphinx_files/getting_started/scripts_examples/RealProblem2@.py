#!/usr/bin/env python
#
# RealProblem2.py

import sys
from pyomo.opt.blackbox import RealOptProblem

class RealProblem2(RealOptProblem):

    def __init__(self):
        RealOptProblem.__init__(self)
        self.nvars=4

    def function_value(self, point):
        return point.vars[0] - point.vars[1] + (point.vars[2]-1.5)**2 + (point.vars[3]+2)**4

problem = RealProblem2()
problem.main(sys.argv)
