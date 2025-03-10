#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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
This model is adapted from Noriyuki Yoshio's model for his and Biegler's
2021 publication in AIChE.


Yoshio, N, Biegler, L.T. Demand-based optimization of a chlorobenzene process
with high-fidelity and surrogate reactor models under trust region strategies.
AIChE J. 2021; 67:e17054. https://doi.org/10.1002/aic.17054
"""

from pyomo.environ import ConcreteModel, Var, ExternalFunction, Objective
from pyomo.opt import SolverFactory


def ext_fcn(a, b):
    return a**2 + b**2


def grad_ext_fcn(args, fixed):
    a, b = args[:2]
    return [2 * a, 2 * b]


def create_model():
    m = ConcreteModel()
    m.name = 'Example 2: Yoshio'

    m.x1 = Var(initialize=0)
    m.x2 = Var(bounds=(-2.0, None), initialize=0)

    m.EF = ExternalFunction(ext_fcn, grad_ext_fcn)

    @m.Constraint()
    def con(m):
        return 2 * m.x1 + m.x2 + 10.0 == m.EF(m.x1, m.x2)

    m.obj = Objective(expr=(m.x1 - 1) ** 2 + (m.x2 - 3) ** 2 + m.EF(m.x1, m.x2) ** 2)
    return m


def basis_rule(component, ef_expr):
    x = ef_expr.arg(0)
    y = ef_expr.arg(1)
    return x**2 - y  # This is the low fidelity model


# This problem takes more than the default maximum iterations (50) to solve.
# In testing (Mac 10.15/ipopt version 3.12.12 from conda),
# it took 70 iterations.
def main():
    m = create_model()
    optTRF = SolverFactory('trustregion', maximum_iterations=100, verbose=True)
    optTRF.solve(m, [m.x1], ext_fcn_surrogate_map_rule=basis_rule)


if __name__ == '__main__':
    main()
