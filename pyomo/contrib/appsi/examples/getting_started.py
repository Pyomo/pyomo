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
from pyomo.contrib import appsi
from pyomo.common.timing import HierarchicalTimer


def main(plot=True, n_points=200):
    import numpy as np

    # create a Pyomo model
    m = pyo.ConcreteModel()
    m.x = pyo.Var()
    m.y = pyo.Var()
    m.p = pyo.Param(initialize=1, mutable=True)

    m.obj = pyo.Objective(expr=m.x**2 + m.y**2)
    m.c1 = pyo.Constraint(expr=m.y >= (m.x + 1) ** 2)
    m.c2 = pyo.Constraint(expr=m.y >= (m.x - m.p) ** 2)

    opt = appsi.solvers.Cplex()  # create an APPSI solver interface
    opt.config.load_solution = False  # modify the config options
    # change how automatic updates are handled
    opt.update_config.check_for_new_or_removed_vars = False
    opt.update_config.update_vars = False

    # write a for loop to vary the value of parameter p from 1 to 10
    p_values = [float(i) for i in np.linspace(1, 10, n_points)]
    obj_values = list()
    x_values = list()
    timer = HierarchicalTimer()  # create a timer for some basic profiling
    timer.start('p loop')
    for p_val in p_values:
        m.p.value = p_val
        res = opt.solve(m, timer=timer)
        assert res.termination_condition == appsi.base.TerminationCondition.optimal
        obj_values.append(res.best_feasible_objective)
        opt.load_vars([m.x])
        x_values.append(m.x.value)
    timer.stop('p loop')
    print(timer)

    if plot:
        import matplotlib.pyplot as plt

        # plot the results
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('p')
        ax1.set_ylabel('objective')
        ax1.plot(p_values, obj_values, ':k', label='objective')

        ax2 = ax1.twinx()
        ax2.set_ylabel('x')
        ax2.plot(p_values, x_values, '-b', label='x')

        fig.legend()
        plt.show()


if __name__ == '__main__':
    main()
