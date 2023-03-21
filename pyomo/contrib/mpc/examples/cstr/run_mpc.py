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

import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from pyomo.contrib.mpc.examples.cstr.model import (
    create_instance,
    _plot_time_indexed_variables,
    _step_time_indexed_variables,
)


def get_steady_state_data(target, tee=False):
    m = create_instance(dynamic=False)
    interface = mpc.DynamicModelInterface(m, m.time)
    var_set, tr_cost = interface.get_penalty_from_target(target)
    m.target_set = var_set
    m.tracking_cost = tr_cost
    m.objective = pyo.Objective(expr=sum(m.tracking_cost[:, 0]))
    m.flow_in[:].unfix()
    solver = pyo.SolverFactory("ipopt")
    solver.solve(m, tee=tee)
    return interface.get_data_at_time(0)


def run_cstr_mpc(
    initial_data,
    setpoint_data,
    samples_per_controller_horizon=5,
    sample_time=2.0,
    ntfe_per_sample_controller=2,
    ntfe_plant=5,
    simulation_steps=5,
    tee=False,
):
    controller_horizon = sample_time * samples_per_controller_horizon
    ntfe = ntfe_per_sample_controller * samples_per_controller_horizon
    m_controller = create_instance(horizon=controller_horizon, ntfe=ntfe)
    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.time)
    t0_controller = m_controller.time.first()

    m_plant = create_instance(horizon=sample_time, ntfe=ntfe_plant)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.time)

    # Sets initial conditions and initializes
    controller_interface.load_data(initial_data)
    plant_interface.load_data(initial_data)

    #
    # Add objective to controller model
    #
    setpoint_variables = [m_controller.conc[:, "A"], m_controller.conc[:, "B"]]
    vset, tr_cost = controller_interface.get_penalty_from_target(
        setpoint_data, variables=setpoint_variables
    )
    m_controller.setpoint_set = vset
    m_controller.tracking_cost = tr_cost
    m_controller.objective = pyo.Objective(
        expr=sum(
            m_controller.tracking_cost[i, t]
            for i in m_controller.setpoint_set
            for t in m_controller.time
            if t != m_controller.time.first()
        )
    )

    #
    # Unfix input in controller model
    #
    m_controller.flow_in[:].unfix()
    m_controller.flow_in[t0_controller].fix()
    sample_points = [i * sample_time for i in range(samples_per_controller_horizon + 1)]
    input_set, pwc_con = controller_interface.get_piecewise_constant_constraints(
        [m_controller.flow_in], sample_points
    )
    m_controller.input_set = input_set
    m_controller.pwc_con = pwc_con

    sim_t0 = 0.0

    #
    # Initialize data structure to hold results of "rolling horizon"
    # simulation.
    #
    sim_data = plant_interface.get_data_at_time([sim_t0])

    solver = pyo.SolverFactory("ipopt")
    non_initial_plant_time = list(m_plant.time)[1:]
    ts = sample_time + t0_controller
    for i in range(simulation_steps):
        # The starting point of this part of the simulation
        # in "real" time (rather than the model's time set)
        sim_t0 = i * sample_time

        #
        # Solve controller model to get inputs
        #
        res = solver.solve(m_controller, tee=tee)
        pyo.assert_optimal_termination(res)
        ts_data = controller_interface.get_data_at_time(ts)
        input_data = ts_data.extract_variables([m_controller.flow_in])

        plant_interface.load_data(input_data)

        #
        # Solve plant model to simulate
        #
        res = solver.solve(m_plant, tee=tee)
        pyo.assert_optimal_termination(res)

        #
        # Extract data from simulated model
        #
        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0 - m_plant.time.first())
        sim_data.concatenate(m_data)

        #
        # Re-initialize plant model
        #
        tf_data = plant_interface.get_data_at_time(m_plant.time.last())
        plant_interface.load_data(tf_data)

        #
        # Re-initialize controller model
        #
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_controller)

    return m_plant, sim_data


def main():
    init_steady_target = mpc.ScalarData({"flow_in[*]": 0.3})
    init_data = get_steady_state_data(init_steady_target, tee=False)
    setpoint_target = mpc.ScalarData({"flow_in[*]": 1.2})
    setpoint_data = get_steady_state_data(setpoint_target, tee=False)

    m, sim_data = run_cstr_mpc(init_data, setpoint_data, tee=False)

    _plot_time_indexed_variables(sim_data, [m.conc[:, "A"], m.conc[:, "B"]], show=True)
    _step_time_indexed_variables(sim_data, [m.flow_in[:]], show=True)


if __name__ == "__main__":
    main()
