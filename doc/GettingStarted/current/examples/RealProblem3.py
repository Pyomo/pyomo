#!/usr/bin/env python
#
# RealProblem3.py

import sys
from pyomo.opt.blackbox import RealOptProblem, response_enum

# @prob:
class RealProblem3(RealOptProblem):

    def __init__(self):
        RealOptProblem.__init__(self)
        self.nvars=4
        self.ncons=4
        self.response_types = [response_enum.FunctionValue, 
                                response_enum.Gradient,
                                response_enum.Hessian, 
                                response_enum.NonlinearConstraintValues,
                                response_enum.Jacobian]

    def function_value(self, point):
        return point.vars[0] - point.vars[1] + (point.vars[2]-1.5)**2 + (point.vars[3]+2)**4

    def gradient(self, point):
        return [1, -1, 2*(point.vars[2]-1.5), 4*(point.vars[3]+2)**3]

    def hessian(self, point):
        H = []
        H.append( (2,2,2) )
        H.append( (3,3,12*(point.vars[3]+2)**2) )
        return H

    def nonlinear_constraint_values(self, point):
        C = []
        C.append( sum(point.vars) )
        C.append( sum(x**2 for x in point.vars) )
        return C

    def jacobian(self, point):
        J = []
        for j in range(self.nvars):
            J.append( (0,j,1) )
        for j in range(self.nvars):
            J.append( (1,j,2*point.vars[j]) )
        return J
# @:prob

# @main:
problem = RealProblem3()
problem.main(sys.argv)
# @:main
