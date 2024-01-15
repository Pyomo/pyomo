#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from __future__ import division
import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_outputs import (
    ReactorConcentrationsOutputModel,
)


def maximize_cb_outputs(show_solver_log=False):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the concentration of component B coming out of the reactor
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorConcentrationsOutputModel())

    # The reaction rate constants and the feed concentration will
    # be fixed for this example
    m.k1con = pyo.Constraint(expr=m.reactor.inputs['k1'] == 5 / 6)
    m.k2con = pyo.Constraint(expr=m.reactor.inputs['k2'] == 5 / 3)
    m.k3con = pyo.Constraint(expr=m.reactor.inputs['k3'] == 1 / 6000)
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb'], sense=pyo.maximize)

    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


if __name__ == '__main__':
    m = maximize_cb_outputs(show_solver_log=True)
    m.pprint()
