#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#
#  Development of this module was conducted as part of the Institute for
#  the Design of Advanced Energy Systems (IDAES) with support through the
#  Simulation-Based Engineering, Crosscutting Research Program within the
#  U.S. Department of Energy’s Office of Fossil Energy and Carbon Management.
#
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""
This model is an adaptation of Eason & Biegler's example in the
original version of the Trust Region solver.

Eason, J.P. and Biegler, L.T. (2018), Advanced trust region optimization
strategies for glass box/black box models. AIChE J, 64: 3934-3943.
https://doi.org/10.1002/aic.16364
"""

from pyomo.environ import (
    ConcreteModel,
    Var,
    Reals,
    ExternalFunction,
    sin,
    cos,
    sqrt,
    Constraint,
    Objective,
)
from pyomo.opt import SolverFactory


def ext_fcn(a, b):
    return sin(a - b)


def grad_ext_fcn(args, fixed):
    a, b = args[:2]
    return [cos(a - b), -cos(a - b)]


def create_model():
    m = ConcreteModel()
    m.name = 'Example 1: Eason'
    m.z = Var(range(3), domain=Reals, initialize=2.0)
    m.x = Var(range(2), initialize=2.0)
    m.x[1] = 1.0

    m.ext_fcn = ExternalFunction(ext_fcn, grad_ext_fcn)

    m.obj = Objective(
        expr=(m.z[0] - 1.0) ** 2
        + (m.z[0] - m.z[1]) ** 2
        + (m.z[2] - 1.0) ** 2
        + (m.x[0] - 1.0) ** 4
        + (m.x[1] - 1.0) ** 6
    )

    m.c1 = Constraint(
        expr=m.x[0] * m.z[0] ** 2 + m.ext_fcn(m.x[0], m.x[1]) == 2 * sqrt(2.0)
    )
    m.c2 = Constraint(expr=m.z[2] ** 4 * m.z[1] ** 2 + m.z[1] == 8 + sqrt(2.0))
    return m


def main():
    m = create_model()
    optTRF = SolverFactory('trustregion', maximum_iterations=10, verbose=True)
    optTRF.solve(m, [m.z[0], m.z[1], m.z[2]])


if __name__ == '__main__':
    main()
