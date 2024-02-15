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

import random
import sys

import pyomo.kernel as pmo

try:
    import numpy as np
except:
    print("This example requires numpy")
    sys.exit(0)

# Set to True to show 3d plots
show_plots = False


def f(x, y, package=pmo):
    return (
        -20 * package.exp(-2.0 * package.sqrt(0.5 * (x**2 + y**2)))
        - package.exp(0.5 * (package.cos(2 * np.pi * x) + package.cos(2 * np.pi * y)))
        + np.e
        + 20.0
    )


def g(x, y, package=pmo):
    return (x - 3) ** 2 + (y - 1) ** 2


m = pmo.block()
m.x = pmo.variable(lb=-5, ub=5)
m.y = pmo.variable(lb=-5, ub=5)
m.z = pmo.variable()
m.obj = pmo.objective(m.z)

m.real = pmo.block()
m.real.f = pmo.constraint(m.z >= f(m.x, m.y))
m.real.g = pmo.constraint(m.z == g(m.x, m.y))

#
# Approximate f and g using piecewise-linear functions
#

m.approx = pmo.block()

tri = pmo.piecewise_util.generate_delaunay([m.x, m.y], num=25)
pw_xarray, pw_yarray = np.transpose(tri.points)

fvals = f(pw_xarray, pw_yarray, package=np)
pw_f = pmo.piecewise_nd(tri, fvals, input=[m.x, m.y], output=m.z, bound='lb')
m.approx.pw_f = pw_f

gvals = g(pw_xarray, pw_yarray, package=np)
pw_g = pmo.piecewise_nd(tri, gvals, input=[m.x, m.y], output=m.z, bound='eq')
m.approx.pw_g = pw_g

#
# Solve the approximate model to generate a warmstart for
# the real model
#

m.real.deactivate()
m.approx.activate()
glpk = pmo.SolverFactory("glpk")
status = glpk.solve(m)
assert str(status.solver.status) == "ok"
assert str(status.solver.termination_condition) == "optimal"

print("Approximate f value at MIP solution: %s" % (pw_f((m.x.value, m.y.value))))
print("Approximate g value at MIP solution: %s" % (pw_g((m.x.value, m.y.value))))
print("Real f value at MIP solution: %s" % (f(m.x.value, m.y.value)))
print("Real g value at MIP solution: %s" % (g(m.x.value, m.y.value)))

#
# Solve the real nonlinear model using a local solver
#

m.real.activate()
m.approx.deactivate()
ipopt = pmo.SolverFactory("ipopt")
status = ipopt.solve(m)
assert str(status.solver.status) == "ok"
assert str(status.solver.termination_condition) == "optimal"
print("Real f value at NL solution: %s" % (f(m.x.value, m.y.value)))
print("Real g value at NL solution: %s" % (f(m.x.value, m.y.value)))

if show_plots:
    import matplotlib.pylab as plt
    import mpl_toolkits.mplot3d

    #
    # Plot the approximation of f
    #

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(pw_xarray, pw_yarray, fvals, color='yellow', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('f approximation')

    #
    # Plot the approximation of g
    #

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(pw_xarray, pw_yarray, gvals, color='blue', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('g approximation')

    #
    # Plot the solution of the nonlinear model
    #

    xarray = np.arange(m.x.lb, m.x.ub, 0.01)
    yarray = np.arange(m.y.lb, m.y.ub, 0.01)
    xarray, yarray = np.meshgrid(xarray, yarray)
    fvals = f(xarray, yarray, package=np)
    gvals = g(xarray, yarray, package=np)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(m.x.value, m.y.value, f(m.x.value, m.y.value), color='black', s=2**6)
    ax.plot_surface(xarray, yarray, fvals, linewidth=0, cmap=plt.cm.jet, alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.plot_surface(xarray, yarray, gvals, linewidth=0, cmap=plt.cm.jet, alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('NL solution')

    plt.show()
