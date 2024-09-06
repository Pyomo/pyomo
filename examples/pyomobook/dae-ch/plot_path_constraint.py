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


# @plot_path:
def plotter(subplot, x, *y, **kwds):
    plt.subplot(subplot)
    for i, _y in enumerate(y):
        plt.plot(list(x), [value(_y[t]) for t in x], 'brgcmk'[i % 6])
        if kwds.get('points', False):
            plt.plot(list(x), [value(_y[t]) for t in x], 'o')
    plt.title(kwds.get('title', ''))
    plt.legend(tuple(_y.name for _y in y))
    plt.xlabel(x.name)


import matplotlib.pyplot as plt

plotter(121, m.t, m.x1, m.x2, title='Differential Variables')
plotter(122, m.t, m.u, title='Control Variable', points=True)
plt.show()
# @:plot_path
