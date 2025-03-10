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
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.common.dependencies.matplotlib import pyplot as plt


def _flow_eqn_rule(m, t):
    return m.flow_in[t] - m.flow_out[t] == 0


def _conc_out_eqn_rule(m, t, j):
    return m.conc[t, j] - m.conc_out[t, j] == 0


def _rate_eqn_rule(m, t, j):
    return m.rate_gen[t, j] - m.stoich[j] * m.k_rxn * m.conc[t, "A"] == 0


def _conc_diff_eqn_rule(m, t, j):
    return (
        m.dcdt[t, j]
        - (
            m.flow_in[t] * m.conc_in[t, j]
            - m.flow_out[t] * m.conc_out[t, j]
            + m.rate_gen[t, j]
        )
        == 0
    )


def _conc_steady_eqn_rule(m, t, j):
    return (
        m.flow_in[t] * m.conc_in[t, j]
        - m.flow_out[t] * m.conc_out[t, j]
        + m.rate_gen[t, j]
    ) == 0


def make_model(dynamic=True, horizon=10.0):
    m = pyo.ConcreteModel()
    m.comp = pyo.Set(initialize=["A", "B"])
    if dynamic:
        m.time = dae.ContinuousSet(initialize=[0, horizon])
    else:
        m.time = pyo.Set(initialize=[0])
    time = m.time
    comp = m.comp

    m.stoich = pyo.Param(m.comp, initialize={"A": -1, "B": 1}, mutable=True)
    m.k_rxn = pyo.Param(initialize=1.0, mutable=True)

    m.conc = pyo.Var(m.time, m.comp)
    if dynamic:
        m.dcdt = dae.DerivativeVar(m.conc, wrt=m.time)

    m.flow_in = pyo.Var(time, bounds=(0, None))
    m.flow_out = pyo.Var(time, bounds=(0, None))
    m.flow_eqn = pyo.Constraint(time, rule=_flow_eqn_rule)

    m.conc_in = pyo.Var(time, comp, bounds=(0, None))
    m.conc_out = pyo.Var(time, comp, bounds=(0, None))
    m.conc_out_eqn = pyo.Constraint(time, comp, rule=_conc_out_eqn_rule)

    m.rate_gen = pyo.Var(time, comp)
    m.rate_eqn = pyo.Constraint(time, comp, rule=_rate_eqn_rule)

    if dynamic:
        m.conc_diff_eqn = pyo.Constraint(time, comp, rule=_conc_diff_eqn_rule)
    else:
        m.conc_steady_eqn = pyo.Constraint(time, comp, rule=_conc_steady_eqn_rule)

    return m


def initialize_model(m, dynamic=True, ntfe=None):
    if ntfe is not None and not dynamic:
        raise RuntimeError("Cannot provide ntfe to initialize steady model")
    elif dynamic and ntfe is None:
        ntfe = 10
    if dynamic:
        disc = pyo.TransformationFactory("dae.finite_difference")
        disc.apply_to(m, wrt=m.time, nfe=ntfe, scheme="BACKWARD")

    t0 = m.time.first()

    # Fix inputs
    m.conc_in[:, "A"].fix(5.0)
    m.conc_in[:, "B"].fix(0.01)
    m.flow_in[:].fix(1.0)
    m.flow_in[t0].fix(0.1)

    if dynamic:
        # Fix initial conditions if dynamic
        m.conc[t0, "A"].fix(1.0)
        m.conc[t0, "B"].fix(0.0)


def create_instance(dynamic=True, horizon=None, ntfe=None):
    if horizon is None and dynamic:
        horizon = 10.0
    if ntfe is None and dynamic:
        ntfe = 10
    m = make_model(horizon=horizon, dynamic=dynamic)
    initialize_model(m, ntfe=ntfe, dynamic=dynamic)
    return m


def _plot_time_indexed_variables(
    data, keys, show=False, save=False, fname=None, transparent=False
):
    fig, ax = plt.subplots()
    time = data.get_time_points()
    for i, key in enumerate(keys):
        data_list = data.get_data_from_key(key)
        label = str(data.get_cuid(key))
        ax.plot(time, data_list, label=label)
    ax.legend()

    if show:
        plt.show()
    if save:
        if fname is None:
            fname = "states.png"
        fig.savefig(fname, transparent=transparent)

    return fig, ax


def _step_time_indexed_variables(
    data, keys, show=False, save=False, fname=None, transparent=False
):
    fig, ax = plt.subplots()
    time = data.get_time_points()
    for i, key in enumerate(keys):
        data_list = data.get_data_from_key(key)
        label = str(data.get_cuid(key))
        ax.step(time, data_list, label=label)
    ax.legend()

    if show:
        plt.show()
    if save:
        if fname is None:
            fname = "inputs.png"
        fig.savefig(fname, transparent=transparent)

    return fig, ax


def main():
    # Make sure steady and dynamic models are square, structurally
    # nonsingular models.
    m_steady = create_instance(dynamic=False)
    steady_igraph = IncidenceGraphInterface(m_steady)
    assert len(steady_igraph.variables) == len(steady_igraph.constraints)
    steady_vdmp, steady_cdmp = steady_igraph.dulmage_mendelsohn()
    assert not steady_vdmp.unmatched and not steady_cdmp.unmatched

    m = create_instance(horizon=100.0, ntfe=100)
    igraph = IncidenceGraphInterface(m)
    assert len(igraph.variables) == len(igraph.constraints)
    vdmp, cdmp = igraph.dulmage_mendelsohn()
    assert not vdmp.unmatched and not cdmp.unmatched


if __name__ == "__main__":
    main()
