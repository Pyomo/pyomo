from __future__ import division

from math import pi

import pyutilib.th as unittest
from pyomo.core import ConcreteModel, Expression, Var, cos, exp, sin, value
from pyomo.core.expr.current import identify_variables
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc


class TestMcCormick(unittest.TestCase):

    def test_trig_functions(self):
        m = ConcreteModel()
        m.x = Var(bounds=(pi / 6, pi / 3), initialize=pi / 4)
        m.e = Expression(expr=cos(pow(m.x, 2)) * sin(pow(m.x, -3)))
        mc_ccVals, mc_cvVals, aff_cc, aff_cv = make2dPlot(m.e.expr, 50)
        self.assertEqual(mc_ccVals[1], 0.6443888590411435)
        self.assertEqual(mc_cvVals[1], 0.2328315489072924)
        self.assertEqual(aff_cc[1], 0.9674274332870583)
        self.assertEqual(aff_cv[1], -1.578938503009686)

    def test_mc_3d(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-2, 1), initialize=-1)
        m.y = Var(bounds=(-1, 2), initialize=0)
        m.e = Expression(expr=m.x * pow(exp(m.x) - m.y, 2))
        ccSurf, cvSurf, ccAffine, cvAffine = make3dPlot(m.e.expr, 30)
        self.assertEqual(ccSurf[48], 11.5655473482574)
        self.assertEqual(cvSurf[48], -15.28102124928224)
        self.assertEqual(ccAffine[48], 11.565547348257398)
        self.assertEqual(cvAffine[48], -23.131094696514797)

    def test_var(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 1), initialize=3)
        mc_var = mc(m.x)
        self.assertEqual(mc_var.lower(), -1)
        self.assertEqual(mc_var.upper(), 1)

    def test_fixed_var(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-50, 80), initialize=3)
        m.y = Var(bounds=(0, 6), initialize=2)
        m.y.fix()
        mc_expr = mc(m.x * m.y)
        self.assertEqual(mc_expr.lower(), -100)
        self.assertEqual(mc_expr.upper(), 160)


def make2dPlot(expr, numticks=10, show_plot=False):
    mc_ccVals = [None] * (numticks + 1)
    mc_cvVals = [None] * (numticks + 1)
    aff_cc = [None] * (numticks + 1)
    aff_cv = [None] * (numticks + 1)
    fvals = [None] * (numticks + 1)
    mc_expr = mc(expr)
    x = next(identify_variables(expr))  # get the first variable
    tick_length = (x.ub - x.lb) / numticks
    xaxis = [x.lb + tick_length * n for n in range(numticks + 1)]

    x_val = value(x)  # initial value of x
    cc = mc_expr.subcc()  # Concave overestimator subgradient at x_val
    cv = mc_expr.subcv()  # Convex underestimator subgradient at x_val
    f_cc = mc_expr.concave()  # Concave overestimator value at x_val
    f_cv = mc_expr.convex()  # Convex underestimator value at x_val
    for i, x_tick in enumerate(xaxis):
        aff_cc[i] = cc[x] * (x_tick - x_val) + f_cc
        aff_cv[i] = cv[x] * (x_tick - x_val) + f_cv
        mc_expr.changePoint(x, x_tick)
        mc_ccVals[i] = mc_expr.concave()
        mc_cvVals[i] = mc_expr.convex()
        fvals[i] = value(expr)
    if show_plot:
        import matplotlib.pyplot as plt
        plt.plot(xaxis, fvals, 'r', xaxis, mc_ccVals, 'b--', xaxis,
                 mc_cvVals, 'b--', xaxis, aff_cc, 'k|', xaxis, aff_cv, 'k|')
        plt.show()
    return mc_ccVals, mc_cvVals, aff_cc, aff_cv


def make3dPlot(expr, numticks=30, show_plot=False):
    ccSurf = [None] * ((numticks + 1)**2)
    cvSurf = [None] * ((numticks + 1)**2)
    fvals = [None] * ((numticks + 1)**2)
    xaxis2d = [None] * ((numticks + 1)**2)
    yaxis2d = [None] * ((numticks + 1)**2)
    ccAffine = [None] * ((numticks + 1)**2)
    cvAffine = [None] * ((numticks + 1)**2)

    eqn = mc(expr)
    vars = identify_variables(expr)
    x = next(vars)
    y = next(vars)
    x_tick_length = (x.ub - x.lb) / numticks
    y_tick_length = (y.ub - y.lb) / numticks
    xaxis = [x.lb + x_tick_length * n for n in range(numticks + 1)]
    yaxis = [y.lb + y_tick_length * n for n in range(numticks + 1)]

    # Making the affine tangent planes
    ccSlope = eqn.subcc()
    cvSlope = eqn.subcv()
    x_val = value(x)
    y_val = value(y)
    f_cc = eqn.concave()
    f_cv = eqn.convex()

    # To Visualize Concave Affine Plane for different points
    for i, x_tick in enumerate(xaxis):
        eqn.changePoint(x, x_tick)
        for j, y_tick in enumerate(yaxis):
            ccAffine[i + (numticks + 1) * j] = (
                ccSlope[x] * (x_tick - x_val) +
                ccSlope[y] * (y_tick - y_val) + f_cc)
            cvAffine[i + (numticks + 1) * j] = (
                cvSlope[x] * (x_tick - x_val) +
                cvSlope[y] * (y_tick - y_val) + f_cv)
            xaxis2d[i + (numticks + 1) * j] = x_tick
            yaxis2d[i + (numticks + 1) * j] = y_tick
            eqn.changePoint(y, y_tick)
            ccSurf[i + (numticks + 1) * j] = eqn.concave()
            cvSurf[i + (numticks + 1) * j] = eqn.convex()
            fvals[i + (numticks + 1) * j] = value(expr)

    if show_plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        assert Axes3D  # silence pyflakes

        # Plotting Solutions in 3D
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(xaxis2d, yaxis2d, cvSurf, color='b')
        ax.scatter(xaxis2d, yaxis2d, fvals, color='r')
        ax.scatter(xaxis2d, yaxis2d, ccSurf, color='b')

        # To Visualize Concave Affine Plane for different points
        ax.scatter(xaxis2d, yaxis2d, cvAffine, color='k')

        # Create a better view
        ax.view_init(10, 270)
        plt.show()

    return ccSurf, cvSurf, ccAffine, cvAffine


if __name__ == '__main__':
    unittest.main()
