from __future__ import division
from pyomo.core.expr.current import identify_variables
import matplotlib.pyplot as plt
%matplotlib inline
from pyomo.environ import *
from pyomo_mcpp import McCormick as mc
import numpy as np
import math

m = ConcreteModel()
m.x = Var(bounds = (math.pi/6, math.pi/3), initialize = math.pi/4)
m.e = Expression(expr = cos(pow(m.x, 2))*sin(pow(m.x,-3)))


def make2dPlot(expr, tickSpace, affinept = None):
    mc_ccVals = []
    mc_cvVals = []
    aff_cc = []
    aff_cv = []
    fvals = []
    eqn = mc(expr)
    x = list(identify_variables(expr))[0]
    xaxis = np.arange(x.lb, x.ub, tickSpace)
    if (affinept != None):
        eqn.changePoint(x, affinept)
        x.set_value(affinept)
    cc = eqn.subcc()
    cv = eqn.subcv()
    for i in xaxis:
        aff_cc += [cc[x]*(i - value(x)) + eqn.concave()]
        aff_cv += [cv[x]*(i - value(x)) + eqn.convex()]
    for i in xaxis:
        eqn.changePoint(x, i)
        mc_ccVals += [eqn.concave()]
        mc_cvVals += [eqn.convex()]
        fvals += [value(expr)]
    plt.plot(xaxis, fvals, 'r', xaxis, mc_ccVals, 'b--', xaxis, mc_cvVals, 'b--', xaxis, aff_cc, 'k|', xaxis, aff_cv, 'k|')
    
make2dPlot(m.e, 0.01)