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

# Heat Conduction Model

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar

m = pyo.ConcreteModel()
m.time = ContinuousSet(bounds=(0, 1))
m.x = ContinuousSet(bounds=(0, 10))
m.y = ContinuousSet(bounds=(0, 5))
m.T = pyo.Var(m.x, m.y, m.time)
m.u = pyo.Var(m.x, m.y, m.time)
m.T0 = pyo.Param(initialize=5)
m.TD = pyo.Param(m.x, m.y, initialize=25)
m.Ux0 = pyo.Param(initialize=10)
m.Uy5 = pyo.Param(initialize=15)

m.dTdx = DerivativeVar(m.T, wrt=m.x)
m.d2Tdx2 = DerivativeVar(m.T, wrt=(m.x, m.x))
m.dTdy = DerivativeVar(m.T, wrt=m.y)
m.d2Tdy2 = DerivativeVar(m.T, wrt=(m.y, m.y))
m.dTdt = DerivativeVar(m.T, wrt=m.time)


def _heateq(m, i, j, k):
    return m.d2Tdx2[i, j, k] + m.d2Tdy2[i, j, k] + m.u[i, j, k] == m.dTdt[i, j, k]


m.heateq = pyo.Constraint(m.x, m.y, m.time, rule=_heateq)


def _initT(m, i, j):
    return m.T[i, j, 0] == m.T0


m.initT = pyo.Constraint(m.x, m.y, rule=_initT)


def _xbound(m, j, k):
    return m.dTdx[0, j, k] == m.Ux0


m.xbound = pyo.Constraint(m.y, m.time, rule=_xbound)


def _ybound(m, i, k):
    return m.dTdy[i, 5, k] == m.Uy5


m.ybound = pyo.Constraint(m.x, m.time, rule=_ybound)

# def _intExp(m,i,j):
#     return m.T[i,j,1] - m.TD[i,j]
# m.intExp = Expression(m.x,m.y,rule=_intExp)

# def _obj(m):
#     return Integral(Integral(expr=m.intExp,wrt=m.x,bounds=(0,10)),
#                     wrt=m.y,bounds=(0,5))
# m.obj = pyo.Objective(rule=_obj)
