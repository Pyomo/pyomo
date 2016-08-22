from pyomo.environ import *
from pyomo.dae import *
from path_constraint import m

# Discretize model using Orthogonal Collocation
# @disc:
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=7,ncp=6,scheme='LAGRANGE-RADAU')
# @:disc
# @reduce:
discretizer.reduce_collocation_points(m,var=m.u,ncp=1,contset=m.t)
# @:reduce

solver=SolverFactory('ipopt')
results = solver.solve(m, tee=True)

def plotter(subplot, x, *y, **kwds):
    plt.subplot(subplot)
    for i,_y in enumerate(y):
        plt.plot(list(x), [value(_y[t]) for t in x], 'brgcmk'[i%6])
        if kwds.get('points', False):
            plt.plot(list(x), [value(_y[t]) for t in x], 'o')
    plt.title(kwds.get('title',''))
    plt.legend(tuple(_y.name() for _y in y))
    plt.xlabel(x.name())

import matplotlib.pyplot as plt
plotter(121, m.t, m.x1, m.x2, title='Differential Variables')
plotter(122, m.t, m.u, title='Control Variable', points=True)
plt.show()
