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

#
# Example from Scipy odeint examples
#
# Second order differential equation for the angle theta of a pendulum
# acted on by gravity with friction. Reformulated as system of ODEs
#
# theta' = omega
# omega' = -b*omega - c*sin(theta)
#
# Example modified to include time-varying values for b and c

from pyomo.environ import *
from pyomo.dae import *
from pyomo.dae.simulator import Simulator


def create_model():
    m = ConcreteModel()

    m.t = ContinuousSet(bounds=(0.0, 20.0))

    def _b_default(m, t):
        if t >= 15:
            return 0.025
        return 0.25

    m.b = Param(m.t, initialize=0.25, default=_b_default)

    def _c_default(m, t):
        if t >= 7:
            return 50
        return 5

    m.c = Param(m.t, initialize=5.0, default=_c_default)

    m.omega = Var(m.t)
    m.theta = Var(m.t)

    m.domegadt = DerivativeVar(m.omega, wrt=m.t)
    m.dthetadt = DerivativeVar(m.theta, wrt=m.t)

    # Setting the initial conditions
    m.omega[0] = 0.0
    m.theta[0] = 3.14 - 0.1

    def _diffeq1(m, t):
        return m.domegadt[t] == -m.b[t] * m.omega[t] - m.c[t] * sin(m.theta[t])

    m.diffeq1 = Constraint(m.t, rule=_diffeq1)

    def _diffeq2(m, t):
        return m.dthetadt[t] == m.omega[t]

    m.diffeq2 = Constraint(m.t, rule=_diffeq2)

    b_profile = {0: 0.25, 15: 0.025}
    c_profile = {0: 5.0, 7: 50}

    m.var_input = Suffix(direction=Suffix.LOCAL)
    m.var_input[m.b] = b_profile
    m.var_input[m.c] = c_profile

    return m


def simulate_model(m):
    if False:
        # Simulate the model using casadi
        sim = Simulator(m, package='casadi')
        tsim, profiles = sim.simulate(
            numpoints=200, integrator='cvodes', varying_inputs=m.var_input
        )
    else:
        # Simulate the model using scipy
        sim = Simulator(m, package='scipy')
        tsim, profiles = sim.simulate(
            numpoints=200, integrator='vode', varying_inputs=m.var_input
        )

    # Discretize model using Orthogonal Collocation
    discretizer = TransformationFactory('dae.collocation')
    discretizer.apply_to(m, nfe=20, ncp=3)

    # Initialize the discretized model using the simulator profiles
    sim.initialize_model()

    return sim, tsim, profiles


def plot_result(m, sim, tsim, profiles):
    import matplotlib.pyplot as plt

    time = list(m.t)
    omega = [value(m.omega[t]) for t in m.t]
    theta = [value(m.theta[t]) for t in m.t]

    varorder = sim.get_variable_order()
    for idx, v in enumerate(varorder):
        plt.plot(tsim, profiles[:, idx], label=v)
    plt.plot(time, omega, 'o', label='omega interp')
    plt.plot(time, theta, 'o', label='theta interp')
    plt.xlabel('t')
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    model = create_model()
    sim, tsim, profiles = simulate_model(model)
    plot_result(model, sim, tsim, profiles)
