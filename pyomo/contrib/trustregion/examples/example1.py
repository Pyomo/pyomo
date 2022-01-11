#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  The Institute for the Design of Advanced Energy Systems Integrated Platform
#  Framework (IDAES IP) was produced under the DOE Institute for the
#  Design of Advanced Energy Systems (IDAES), and is copyright (c) 2018-2021
#  by the software owners: The Regents of the University of California, through
#  Lawrence Berkeley National Laboratory,  National Technology & Engineering
#  Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia University
#  Research Corporation, et al.  All rights reserved.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

# This model is an adaptation of Eason & Biegler's original example in the
# previous version of the Trust Region solver.

from pyomo.environ import (
    ConcreteModel, Var, Reals, ExternalFunction, sin, cos,
    sqrt, Constraint, Objective)
from pyomo.opt import SolverFactory

def ext_fcn(a, b):
   return sin(a - b)

def grad_ext_fcn(args, fixed):
    a, b = args[:2]
    return [ cos(a - b), -cos(a - b) ]

def create_model():
    m = ConcreteModel()
    m.name = 'Example 1: Eason'
    m.z = Var(range(3), domain=Reals, initialize=2.)
    m.x = Var(range(2), initialize=2.)
    m.x[1] = 1.0

    m.ext_fcn = ExternalFunction(ext_fcn, grad_ext_fcn)

    m.obj = Objective(
        expr=(m.z[0]-1.0)**2 + (m.z[0]-m.z[1])**2 + (m.z[2]-1.0)**2 \
           + (m.x[0]-1.0)**4 + (m.x[1]-1.0)**6
    )

    m.c1 = Constraint(
        expr=m.x[0] * m.z[0]**2 + m.ext_fcn(m.x[0], m.x[1]) == 2*sqrt(2.0)
        )
    m.c2 = Constraint(expr=m.z[2]**4 * m.z[1]**2 + m.z[1] == 8+sqrt(2.0))
    return m

def main():
    m = create_model()
    optTRF = SolverFactory('trustregion', maximum_iterations=10, verbose=True)
    optTRF.solve(m, [m.z[0], m.z[1], m.z[2]])


if __name__ == '__main__':
    main()
