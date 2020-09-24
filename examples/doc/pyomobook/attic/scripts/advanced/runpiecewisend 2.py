import random
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from pyomo.environ import *
from pyomo.opt import (SolverFactory,
                       TerminationCondition)

from PiecewiseND import (BuildPiecewiseND,
                         generate_delaunay)

def f(x, y, package=pyomo.core):
    return (-20 * package.exp(
                -2.0 * package.sqrt(0.5 * (x**2 + y**2))) -
            package.exp(
                0.5 * (package.cos(2*np.pi*x) + \
                       package.cos(2*np.pi*y))) + \
            np.e + 20.0)

def g(x, y, package=pyomo.core):
    return (x-3)**2 + (y-1)**2

model = ConcreteModel()
model.x = Var(bounds=(-5, 5))
model.y = Var(bounds=(-5, 5))

# create nonlinear components
model.nonlinear = Block()
model.nonlinear.obj = Objective(expr=f(model.x, model.y))
model.nonlinear.con = Constraint(expr= \
    g(model.x, model.y) == f(model.x, model.y))

# create linearized components
tri = generate_delaunay([model.x, model.y], num=25)
xarray, yarray = np.transpose(tri.points)
fvals = f(xarray, yarray, package=np)
gvals = g(xarray, yarray, package=np)

model.linear = Block()
model.linear.z = Var()
model.linear.obj = Objective(expr=model.linear.z)
model.linear.pw_obj_mesh = BuildPiecewiseND(
    [model.x,model.y],
    model.linear.z,
    tri,
    fvals)

model.linear.pw_con_mesh = BuildPiecewiseND(
    [model.x,model.y],
    model.linear.z,
    tri,
    gvals)

if __name__ == "__main__":
    # solve the nonlinear version with a bad initial guess
    model.linear.deactivate()
    model.x = 4
    model.y = 4
    with SolverFactory("ipopt") as ipopt:
        results = ipopt.solve(model)
    if (results.Solver.termination_condition !=
        TerminationCondition.optimal):
        print("Failed to solve nonlinear model")
        exit(1)
    print("Nonlinear Objective After Bad Initial "
          "Guess: f(%f, %f) = %f"
          % (model.x(), model.y(), model.nonlinear.obj()))

    # solver the linear version of the model
    model.nonlinear.deactivate()
    model.linear.activate()
    with SolverFactory("glpk") as glpk:
        results = glpk.solve(model)
    if (results.Solver.termination_condition !=
        TerminationCondition.optimal):
        print("Failed to solve discretized model")
        exit(1)
    print("Discretized Objective: f(%f, %f) = %f"
          % (model.x(), model.y(), model.linear.obj()))

    # reactivate the nonlinear model form and use current
    # solution as the warmstart for ipopt
    model.linear.deactivate()
    model.nonlinear.activate()
    with SolverFactory("ipopt") as ipopt:
        results = ipopt.solve(model)
    if (results.Solver.termination_condition !=
        TerminationCondition.optimal):
        print("Failed to solve warmstarted nonlinear model")
        exit(1)
    print("Ipopt Objective Using Warmstart: f(%f, %f) = %f"
          % (model.x(), model.y(), model.nonlinear.obj()))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xarray, yarray, fvals,
                    color='yellow', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(xarray, yarray, gvals,
                    color='blue', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    xarray = np.arange(model.x.lb, model.x.ub, 0.01)
    yarray = np.arange(model.y.lb, model.y.ub, 0.01)
    xarray, yarray = np.meshgrid(xarray, yarray)
    fvals = f(xarray, yarray, package=np)
    gvals = g(xarray, yarray, package=np)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(model.x(), model.y(), model.nonlinear.obj(),
               color='black', s=2**6)
    ax.plot_surface(xarray, yarray, fvals,
                    linewidth=0, cmap=cm.jet,
                    alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(model.x(), model.y(), model.nonlinear.obj(),
               color='black', s=2**6)
    ax.plot_surface(xarray, yarray, gvals,
                    linewidth=0, cmap=cm.jet,
                    alpha=0.6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
