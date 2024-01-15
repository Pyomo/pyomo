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
import pyomo.dae as dae


def make_gas_expansion_model(N=2):
    """
    This is the simplest model I could think of that has a
    subsystem with a non-trivial block triangularization.
    Something like a gas (somehow) undergoing a series
    of isentropic expansions.
    """
    m = pyo.ConcreteModel()
    m.streams = pyo.Set(initialize=range(N + 1))
    m.rho = pyo.Var(m.streams, initialize=1)
    m.P = pyo.Var(m.streams, initialize=1)
    m.F = pyo.Var(m.streams, initialize=1)
    m.T = pyo.Var(m.streams, initialize=1)

    m.R = pyo.Param(initialize=8.31)
    m.Q = pyo.Param(m.streams, initialize=1)
    m.gamma = pyo.Param(initialize=1.4 * m.R.value)

    def mbal(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return m.rho[i - 1] * m.F[i - 1] - m.rho[i] * m.F[i] == 0

    m.mbal = pyo.Constraint(m.streams, rule=mbal)

    def ebal(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return (
                m.rho[i - 1] * m.F[i - 1] * m.T[i - 1]
                + m.Q[i]
                - m.rho[i] * m.F[i] * m.T[i]
                == 0
            )

    m.ebal = pyo.Constraint(m.streams, rule=ebal)

    def expansion(m, i):
        if i == 0:
            return pyo.Constraint.Skip
        else:
            return m.P[i] / m.P[i - 1] - (m.rho[i] / m.rho[i - 1]) ** m.gamma == 0

    m.expansion = pyo.Constraint(m.streams, rule=expansion)

    def ideal_gas(m, i):
        return m.P[i] - m.rho[i] * m.R * m.T[i] == 0

    m.ideal_gas = pyo.Constraint(m.streams, rule=ideal_gas)

    return m


def make_dynamic_model(**disc_args):
    # Level control model
    m = pyo.ConcreteModel()
    m.time = dae.ContinuousSet(initialize=[0.0, 10.0])
    m.height = pyo.Var(m.time, initialize=1.0)
    m.flow_in = pyo.Var(m.time, initialize=1.0)
    m.flow_out = pyo.Var(m.time, initialize=0.5)
    m.dhdt = dae.DerivativeVar(m.height, wrt=m.time, initialize=0.0)

    m.area = pyo.Param(initialize=1.0)
    m.flow_const = pyo.Param(initialize=0.5)

    def diff_eqn_rule(m, t):
        return m.area * m.dhdt[t] - (m.flow_in[t] - m.flow_out[t]) == 0

    m.diff_eqn = pyo.Constraint(m.time, rule=diff_eqn_rule)

    def flow_out_rule(m, t):
        return m.flow_out[t] - (m.flow_const * pyo.sqrt(m.height[t])) == 0

    m.flow_out_eqn = pyo.Constraint(m.time, rule=flow_out_rule)

    default_disc_args = {"wrt": m.time, "nfe": 5, "scheme": "BACKWARD"}
    default_disc_args.update(disc_args)

    discretizer = pyo.TransformationFactory("dae.finite_difference")
    discretizer.apply_to(m, **default_disc_args)

    return m


def make_degenerate_solid_phase_model():
    """
    From the solid phase thermo package of a moving bed chemical looping
    combustion reactor. This example was first presented in [1]

    [1] Parker, R. Nonlinear programming strategies for dynamic models of
    chemical looping combustion reactors. Pres. AIChE Annual Meeting, 2020.

    """
    m = pyo.ConcreteModel()
    m.components = pyo.Set(initialize=[1, 2, 3])
    m.x = pyo.Var(m.components, initialize=1 / 3)
    m.flow_comp = pyo.Var(m.components, initialize=10)
    m.flow = pyo.Var(initialize=30)
    m.rho = pyo.Var(initialize=1)

    # These are rough approximations of the relevant equations, with the same
    # incidence.
    m.sum_eqn = pyo.Constraint(expr=sum(m.x[j] for j in m.components) - 1 == 0)
    m.holdup_eqn = pyo.Constraint(
        m.components, expr={j: m.x[j] * m.rho - 1 == 0 for j in m.components}
    )
    m.density_eqn = pyo.Constraint(
        expr=1 / m.rho - sum(1 / m.x[j] for j in m.components) == 0
    )
    m.flow_eqn = pyo.Constraint(
        m.components,
        expr={j: m.x[j] * m.flow - m.flow_comp[j] == 0 for j in m.components},
    )

    return m
