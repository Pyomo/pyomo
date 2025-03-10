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
from pyomo.contrib.pynumero.interfaces.external_grey_box import ExternalGreyBoxBlock
from pyomo.contrib.pynumero.examples.external_grey_box.react_example.reactor_model_residuals import (
    ReactorModel,
    ReactorModelWithHessian,
    ReactorModelNoOutputs,
    ReactorModelScaled,
    create_pyomo_reactor_model,
)


def maximize_cb_ratio_residuals_with_output(
    show_solver_log=False, additional_options={}
):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as outputs
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorModel())

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb_ratio'], sense=pyo.maximize)

    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


def maximize_cb_ratio_residuals_with_hessian_with_output(
    show_solver_log=False, additional_options={}
):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as outputs
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorModelWithHessian())

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb_ratio'], sense=pyo.maximize)

    solver = pyo.SolverFactory('cyipopt')
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


def maximize_cb_ratio_residuals_with_hessian_with_output_pyomo(
    show_solver_log=False, additional_options={}
):
    # this example is the same as the one above, but solves with a pure
    # pyomo model - this is mostly for comparison and testing
    m = create_pyomo_reactor_model()
    solver = pyo.SolverFactory('ipopt')
    for k, v in additional_options.items():
        solver.options[k] = v
    solver.options['linear_solver'] = 'mumps'
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


def maximize_cb_ratio_residuals_with_output_scaling(
    show_solver_log=False, additional_options={}
):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as outputs

    # This example also shows how to do scaling.
    # There are two things to scale, the "Pyomo" variables/constraints,
    # and the external model residuals / output equations.
    # The scaling factors for the external residuals and output equations
    # are set in the derived ExternalGreyBoxModel class (see ReactorModelScaled
    # for this example).
    # The scaling factors for the Pyomo part of the model are set using suffixes
    # - this requires that we declare the scaling suffix on the main model as
    #   shown below.
    # - Then the scaling factors can be set directly on the Pyomo variables
    #   and constraints as also shown below.
    # - Note: In this example, the scaling factors for the input and output
    #   variables from the grey box model are set in the finalize_block_construction
    #   callback (again, see ReactorModelScaled)
    m = pyo.ConcreteModel()

    # declare the scaling suffix on the model
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock(external_model=ReactorModelScaled())

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)
    # set a scaling factor for this constraint - if we had additional pyomo
    # variables, we could set them the same way
    m.scaling_factor[m.cafcon] = 42.0

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    m.obj = pyo.Objective(expr=m.reactor.outputs['cb_ratio'], sense=pyo.maximize)

    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


def maximize_cb_ratio_residuals_with_obj(show_solver_log=False, additional_options={}):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as additional pyomo variables
    # and constraints
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock()
    m.reactor.set_external_model(ReactorModelNoOutputs())

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    u = m.reactor.inputs
    m.obj = pyo.Objective(
        expr=u['cb'] / (u['ca'] + u['cc'] + u['cd']), sense=pyo.maximize
    )

    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


def maximize_cb_ratio_residuals_with_pyomo_variables(
    show_solver_log=False, additional_options={}
):
    # in this simple example, we will use an external grey box model representing
    # a steady-state reactor, and solve for the space velocity that maximizes
    # the ratio of B to the other components coming out of the reactor
    # This example illustrates the use of "equality constraints" or residuals
    # in the external grey box example as well as additional pyomo variables
    # and constraints
    m = pyo.ConcreteModel()

    # create a block to store the external reactor model
    m.reactor = ExternalGreyBoxBlock()
    m.reactor.set_external_model(ReactorModelNoOutputs())

    # add a variable and constraint for the cb ratio
    m.cb_ratio = pyo.Var(initialize=1)
    u = m.reactor.inputs
    m.cb_ratio_con = pyo.Constraint(
        expr=u['cb'] / (u['ca'] + u['cc'] + u['cd']) - m.cb_ratio == 0
    )

    # The feed concentration will be fixed for this example
    m.cafcon = pyo.Constraint(expr=m.reactor.inputs['caf'] == 10000)

    # add an objective function that maximizes the concentration
    # of cb coming out of the reactor
    m.obj = pyo.Objective(expr=m.cb_ratio, sense=pyo.maximize)

    solver = pyo.SolverFactory('cyipopt')
    solver.config.options['hessian_approximation'] = 'limited-memory'
    for k, v in additional_options.items():
        solver.config.options[k] = v
    results = solver.solve(m, tee=show_solver_log)
    pyo.assert_optimal_termination(results)
    return m


if __name__ == '__main__':
    m = maximize_cb_ratio_residuals_with_output(show_solver_log=True)

    # the next two are the same model with pyomo/ipopt and external/cyipopt
    m = maximize_cb_ratio_residuals_with_hessian_with_output_pyomo(show_solver_log=True)
    m = maximize_cb_ratio_residuals_with_hessian_with_output(show_solver_log=True)

    aoptions = {
        'hessian_approximation': 'limited-memory',
        #'limited_memory_update_type': 'sr1',
        'nlp_scaling_method': 'user-scaling',
        'print_level': 10,
    }
    m = maximize_cb_ratio_residuals_with_output_scaling(
        show_solver_log=True, additional_options=aoptions
    )

    m = maximize_cb_ratio_residuals_with_obj(show_solver_log=True)

    m = maximize_cb_ratio_residuals_with_pyomo_variables(show_solver_log=True)
