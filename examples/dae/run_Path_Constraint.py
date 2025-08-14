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

import pyomo.environ as pyo
from pyomo.dae import ContinuousSet, DerivativeVar
from Path_Constraint import m

# Discretize model using Finite Difference Method
# discretizer = pyo.TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=20,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation
discretizer = pyo.TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=7, ncp=6, scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

results = pyo.SolverFactory('ipopt').solve(m, tee=True)


def plotter(subplot, x, *series, **kwds):
    plt.subplot(subplot)
    for i, y in enumerate(series):
        plt.plot(
            list(x),
            [pyo.value(y[t]) for t in x],
            'brgcmk'[i % 6] + kwds.get('points', ''),
        )
    plt.title(kwds.get('title', ''))
    plt.legend(tuple(y.name for y in series), frameon=True, edgecolor='k').draw_frame(
        True
    )
    plt.xlabel(x.name)
    plt.gca().set_xlim([0, 1])


import matplotlib.pyplot as plt

plotter(121, m.t, m.x1, m.x2, m.x3, title='Differential Variables')
plotter(122, m.t, m.u, title='Control Variables', points='o-')
plt.show()
