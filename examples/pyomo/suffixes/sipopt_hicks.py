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

# Author: Hans Pirnay, 2012-11-27
# This is the Hicks reactor example from the sIPOPT paper
#
# This Pyomo example is formulated as a python script.
# To run this script execute the following command:
#
# $ pyomo_python sipopt_hicks.py
#
# Execution of this script requires that the ipopt_sens
# solver (distributed with Ipopt) is in the current search
# path for executables on this system. Optionally required
# are the numpy and matplotlib python modules (needed for
# viewing results).

import pyomo.environ
from pyomo.core import *
from pyomo.opt import SolverFactory

### Create the ipopt_sens solver plugin using the ASL interface
solver = 'ipopt_sens'
solver_io = 'nl'
stream_solver = False  # True prints solver output to screen
keepfiles = False  # True prints intermediate file names (.nl,.sol,...)
opt = SolverFactory(solver, solver_io=solver_io)
###

if opt is None:
    print("")
    print("ERROR: Unable to create solver plugin for 'ipopt_sens'")
    print("")
    exit(1)

### model definition
# parameters
nfe = 40
ncp = 3
FE = range(nfe)
CP = range(ncp)

time = 9.0
jj = 100.0
cf = 7.6
alpha = 1.95e-4
tf = 300.0
k10 = 300.0
tc = 290.0
n = 5.0
alpha1 = 1.0e6
alpha2 = 2.0e3
alpha3 = 1.0e-3
c_des = 0.0944
t_des = 0.7766
u_des = 340.0
c_init = 0.1367
t_init = 0.7293
u_init = 390.0
theta = 20.0
yc = tc / (jj * cf)
yf = tf / (jj * cf)

model = ConcreteModel()
model.c_init_var = Var()
model.t_init_var = Var()
model.cdot = Var(FE, CP)
model.tdot = Var(FE, CP)


def c_init_rule(m, i, j):
    return float(i) / nfe * (c_des - c_init) + c_init


model.c = Var(FE, CP, within=NonNegativeReals, initialize=c_init_rule)


def t_init_rule(m, i, j):
    return float(i) / nfe * (t_des - t_init) + t_init


model.t = Var(FE, CP, within=NonNegativeReals, initialize=t_init_rule)
model.u = Var(FE, within=NonNegativeReals, initialize=1.0)

a_init = {}
a_init[0, 0] = 0.19681547722366
a_init[0, 1] = 0.39442431473909
a_init[0, 2] = 0.37640306270047
a_init[1, 0] = -0.06553542585020
a_init[1, 1] = 0.29207341166523
a_init[1, 2] = 0.51248582618842
a_init[2, 0] = 0.02377097434822
a_init[2, 1] = -0.04154875212600
a_init[2, 2] = 0.11111111111111

model.a = Param(FE, CP, initialize=a_init)

h = [1.0 / nfe] * nfe


def cdot_ode_rule(m, i, j):
    return (
        m.cdot[i, j]
        == (1.0 - m.c[i, j]) / theta - k10 * exp(-n / m.t[i, j]) * m.c[i, j]
    )


model.cdot_ode = Constraint(FE, CP, rule=cdot_ode_rule)


def tdot_ode_rule(m, i, j):
    return m.tdot[i, j] == (yf - m.t[i, j]) / theta + k10 * exp(-n / m.t[i, j]) * m.c[
        i, j
    ] - alpha * m.u[i] * (m.t[i, j] - yc)


model.tdot_ode = Constraint(FE, CP, rule=tdot_ode_rule)


def fecolc_rule(m, i, j):
    if i == 0:
        return m.c[i, j] == m.c_init_var + time * h[i] * sum(
            m.a[k, j] * m.cdot[i, k] for k in CP
        )
    else:
        return m.c[i, j] == m.c[i - 1, ncp - 1] + time * h[i] * sum(
            m.a[k, j] * m.cdot[i, k] for k in CP
        )


model.fecolc = Constraint(FE, CP, rule=fecolc_rule)
model.c_init_def = Constraint(expr=model.c_init_var == c_init)
model.t_init_def = Constraint(expr=model.t_init_var == t_init)


def fecolt_rule(m, i, j):
    if i == 0:
        return m.t[i, j] == m.t_init_var + time * h[i] * sum(
            m.a[k, j] * m.tdot[i, k] for k in CP
        )
    else:
        return m.t[i, j] == m.t[i - 1, ncp - 1] + time * h[i] * sum(
            m.a[k, j] * m.tdot[i, k] for k in CP
        )


model.fecolt = Constraint(FE, CP, rule=fecolt_rule)


def obj_rule(m):
    return sum(
        h[i]
        * sum(
            (
                alpha1 * (m.c[i, j] - c_des) ** 2
                + alpha2 * (m.t[i, j] - t_des) ** 2
                + alpha3 * (m.u[i] - u_des) ** 2
            )
            * m.a[j, ncp - 1]
            for j in CP
        )
        for i in range(2, nfe)
    ) + h[0] * sum(
        (
            alpha1
            * (
                (m.c_init_var + time * h[0] * sum(m.a[k, j] * m.cdot[0, k] for k in CP))
                - c_des
            )
            ** 2
            + alpha2
            * (
                (m.t_init_var + time * h[0] * sum(m.a[k, j] * m.tdot[0, k] for k in CP))
                - t_des
            )
            ** 2
            + alpha3 * (m.u[0] - u_des) ** 2
        )
        * m.a[j, ncp - 1]
        for j in CP
    )


model.cost = Objective(rule=obj_rule)
###

### declare suffixes
model.sens_state_0 = Suffix(direction=Suffix.EXPORT)
model.sens_state_1 = Suffix(direction=Suffix.EXPORT)
model.sens_state_value_1 = Suffix(direction=Suffix.EXPORT)
model.sens_sol_state_1 = Suffix(direction=Suffix.IMPORT)
model.sens_init_constr = Suffix(direction=Suffix.EXPORT)
###

### set sIPOPT data
opt.options['run_sens'] = 'yes'
model.sens_state_0[model.c_init_var] = 1
model.sens_state_0[model.t_init_var] = 2
model.sens_state_1[model.c[4, 0]] = 1
model.sens_state_1[model.t[4, 0]] = 2
model.sens_state_value_1[model.c[4, 0]] = 0.135
model.sens_state_value_1[model.t[4, 0]] = 0.745
model.sens_init_constr[model.c_init_def] = 1
model.sens_init_constr[model.t_init_def] = 1
###

### Send the model to ipopt_sens and collect the solution
results = opt.solve(model, keepfiles=keepfiles, tee=stream_solver)
###

# Plot the results
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("")
    print("ERROR: numpy and matplotlib are required to view the example results")
    print("")
    exit(1)


def collocation_points(n_fe, n_cp, h):
    t = 0.0
    r1 = 0.15505102572168
    r2 = 0.64494897427832
    r3 = 1.0
    for i in range(n_fe):
        yield t + h[i] * r1
        yield t + h[i] * r2
        yield t + h[i] * r3
        t += h[i]


def collocation_idx(n_fe, n_cp):
    for i in range(n_fe):
        yield i, 0
        yield i, 1
        yield i, 2


times = np.array([i for i in collocation_points(nfe, ncp, h)])
cnominal = np.zeros((nfe * ncp, 1))
cperturbed = np.zeros((nfe * ncp, 1))
tnominal = np.zeros((nfe * ncp, 1))
tperturbed = np.zeros((nfe * ncp, 1))
for k, (i, j) in enumerate(collocation_idx(nfe, ncp)):
    cnominal[k] = value(model.c[i, j])
    tnominal[k] = value(model.t[i, j])
    cperturbed[k] = value(model.sens_sol_state_1[model.c[i, j]])
    tperturbed[k] = value(model.sens_sol_state_1[model.t[i, j]])

plt.subplot(2, 1, 1)
plt.plot(times, cnominal, label='c_nominal')
# plt.hold(True)
plt.plot(times, cperturbed, label='c_perturbed')
plt.xlim([min(times), max(times)])
plt.legend(loc=0)
plt.subplot(2, 1, 2)
plt.plot(times, tnominal, label='t_nominal')
plt.plot(times, tperturbed, label='t_perturbed')
plt.xlim([min(times), max(times)])
plt.legend(loc=0)
plt.show()
