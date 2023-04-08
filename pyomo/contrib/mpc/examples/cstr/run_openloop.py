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


def get_input_sequence():
    input_sequence = mpc.TimeSeriesData(
        {"flow_in[*]": [0.1, 1.0, 0.5, 1.3, 1.0, 0.3]}, [0.0, 2.0, 4.0, 6.0, 8.0, 15.0]
    )
    return mpc.data.convert.series_to_interval(input_sequence)


def run_cstr_openloop(
    inputs, model_horizon=1.0, ntfe=10, simulation_steps=15, tee=False
):
    m = create_instance(horizon=model_horizon, ntfe=ntfe)
    dynamic_interface = mpc.DynamicModelInterface(m, m.time)

    sim_t0 = 0.0

    #
    # Initialize data structure to hold results of "rolling horizon"
    # simulation.
    #
    sim_data = dynamic_interface.get_data_at_time([sim_t0])

    solver = pyo.SolverFactory("ipopt")
    non_initial_model_time = list(m.time)[1:]
    for i in range(simulation_steps):
        # The starting point of this part of the simulation
        # in "real" time (rather than the model's time set)
        sim_t0 = i * model_horizon

        sim_time = [sim_t0 + t for t in m.time]
        new_inputs = mpc.data.convert.interval_to_series(inputs, time_points=sim_time)
        new_inputs.shift_time_points(m.time.first() - sim_t0)
        dynamic_interface.load_data(new_inputs, tolerance=1e-6)

        #
        # Solve square model to simulate
        #
        res = solver.solve(m, tee=tee)
        pyo.assert_optimal_termination(res)

        #
        # Extract data from simulated model
        #
        m_data = dynamic_interface.get_data_at_time(non_initial_model_time)
        m_data.shift_time_points(sim_t0 - m.time.first())
        sim_data.concatenate(m_data)

        #
        # Re-initialize (initial conditions and variable values)
        #
        # The default is to load this ScalarData at all points in the
        # model's time set.
        tf_data = dynamic_interface.get_data_at_time(m.time.last())
        dynamic_interface.load_data(tf_data)

    return m, sim_data


def main():
    input_sequence = get_input_sequence()
    m, sim_data = run_cstr_openloop(input_sequence, tee=False)
    _plot_time_indexed_variables(sim_data, [m.conc[:, "A"], m.conc[:, "B"]], show=True)
    _step_time_indexed_variables(sim_data, [m.flow_in[:]], show=True)


if __name__ == "__main__":
    main()
