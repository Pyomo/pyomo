#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pyomo.environ as pyo
from pyomo.common.timing import TicTocTimer
from pyomo.core.expr.numeric_expr import LinearExpression

N1 = 10
N2 = 100000

m = pyo.ConcreteModel()
m.x = pyo.Var(list(range(N1)))

timer = TicTocTimer()
timer.tic()

for i in range(N2):
    e = sum(i * m.x[i] for i in range(N1))
timer.toc('created expression with sum function')

for i in range(N2):
    coefs = [i for i in range(N1)]
    lin_vars = [m.x[i] for i in range(N1)]
    e = LinearExpression(constant=0, linear_coefs=coefs, linear_vars=lin_vars)
timer.toc('created expression with LinearExpression constructor')
