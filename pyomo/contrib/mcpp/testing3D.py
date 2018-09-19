from __future__ import division
from pyomo.core.expr.current import identify_variables
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
%matplotlib inline
from pyomo.environ import *
from pyomo_mcpp import McCormick as mc
import numpy as np
import math

m = ConcreteModel()
m.x = Var(bounds = (-2, 1), initialize = -1)
m.y = Var(bounds = (-1, 2), initialize = 0)
m.e = Expression(expr = m.x*pow(exp(m.x)-m.y,2))

def make3dPlot(expr, space, affineptx = None, affinepty = None):
    ccSurf = []
    cvSurf = []
    fvals = []
    xaxis2 = []
    yaxis2 = []
    ccAffine = []
    
    eqn = mc(expr)
    x = list(identify_variables(expr))[0]
    y = list(identify_variables(expr))[1]
    xaxis = np.linspace(x.lb, x.ub, space)
    yaxis = np.linspace(y.lb, y.ub, space)
    
    if (affineptx != None):
        eqn.changePoint(x, affineptx)
        x.set_value(affineptx)
    if (affinepty != None):
        eqn.changePoint(y, affinepty)
        y.set_value(affinepty)

    #Making the affine tangent planes
    ccSlope = eqn.subcc()
    cvSlope = eqn.subcv()
    ccTanPlane = ccSlope[x]*(xaxis - value(x)) + ccSlope[y]*(yaxis - value(y)) + eqn.concave()
    cvTanPlane = cvSlope[x]*(xaxis - value(x)) + cvSlope[y]*(yaxis - value(y)) + eqn.convex()
    
    #To Visualize Concave Affine Plane for different points
    for i in xaxis:
        for j in yaxis:
            ccAffine += [ccSlope[x]*(i - value(x)) + ccSlope[y]*(j - value(y)) + eqn.concave()]
    
    #Making the fxn surface (fvals) and cc and cv surfaces
    for i in xaxis:
        eqn.changePoint(x, i)
        for j in yaxis:
            xaxis2 += [i]
            yaxis2 += [j]
            eqn.changePoint(y, j)
            ccSurf += [eqn.concave()]
            cvSurf += [eqn.convex()]
            fvals += [value(expr)]
            
    #Plotting Solutions in 3D
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    xaxis, yaxis = np.meshgrid(xaxis, yaxis)
    ax.scatter(xaxis2, yaxis2, cvSurf, color = 'b')
    ax.scatter(xaxis2, yaxis2, fvals, color = 'r')
    ax.scatter(xaxis2, yaxis2, ccSurf, color = 'b')
    #ax.plot_surface(xaxis, yaxis, ccTanPlane, color = 'k')
    ax.plot_surface(xaxis, yaxis, cvTanPlane, color = 'k')
    
    #To Visualize Concave Affine Plane for different points
    ax.scatter(xaxis2, yaxis2, ccAffine, color = 'k')
    
    #Create a better view
    ax.view_init(10,270)
    
make3dPlot(m.e, 30)