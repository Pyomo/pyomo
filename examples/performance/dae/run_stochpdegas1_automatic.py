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

import time

import pyomo.environ as pyo
from stochpdegas1_automatic import model

start = time.time()
instance = model.create_instance('stochpdegas_automatic.dat')

# discretize model
discretizer = pyo.TransformationFactory('dae.finite_difference')
discretizer.apply_to(instance, nfe=1, wrt=instance.DIS, scheme='FORWARD')
discretizer.apply_to(instance, nfe=47, wrt=instance.TIME, scheme='BACKWARD')

# What it should be to match description in paper
# discretizer.apply_to(instance,nfe=48,wrt=instance.TIME,scheme='BACKWARD')

TimeStep = instance.TIME[2] - instance.TIME[1]


def supcost_rule(m, k):
    return sum(
        m.cs * m.s[k, j, t] * (TimeStep)
        for j in m.SUP
        for t in m.TIME.get_finite_elements()
    )


instance.supcost = pyo.Expression(instance.SCEN, rule=supcost_rule)


def boostcost_rule(m, k):
    return sum(
        m.ce * m.pow[k, j, t] * (TimeStep)
        for j in m.LINK_A
        for t in m.TIME.get_finite_elements()
    )


instance.boostcost = pyo.Expression(instance.SCEN, rule=boostcost_rule)


def trackcost_rule(m, k):
    return sum(
        m.cd * (m.dem[k, j, t] - m.stochd[k, j, t]) ** 2.0
        for j in m.DEM
        for t in m.TIME.get_finite_elements()
    )


instance.trackcost = pyo.Expression(instance.SCEN, rule=trackcost_rule)


def sspcost_rule(m, k):
    return sum(
        m.cT * (m.px[k, i, m.TIME.last(), j] - m.px[k, i, m.TIME.first(), j]) ** 2.0
        for i in m.LINK
        for j in m.DIS
    )


instance.sspcost = pyo.Expression(instance.SCEN, rule=sspcost_rule)


def ssfcost_rule(m, k):
    return sum(
        m.cT * (m.fx[k, i, m.TIME.last(), j] - m.fx[k, i, m.TIME.first(), j]) ** 2.0
        for i in m.LINK
        for j in m.DIS
    )


instance.ssfcost = pyo.Expression(instance.SCEN, rule=ssfcost_rule)


def cost_rule(m, k):
    return 1e-6 * (
        m.supcost[k] + m.boostcost[k] + m.trackcost[k] + m.sspcost[k] + m.ssfcost[k]
    )


instance.cost = pyo.Expression(instance.SCEN, rule=cost_rule)


def mcost_rule(m):
    return (1.0 / m.S) * sum(m.cost[k] for k in m.SCEN)


instance.mcost = pyo.Expression(rule=mcost_rule)


def eqcvar_rule(m, k):
    return m.cost[k] - m.nu <= m.phi[k]


instance.eqcvar = pyo.Constraint(instance.SCEN, rule=eqcvar_rule)


def obj_rule(m):
    return (1.0 - m.cvar_lambda) * m.mcost + m.cvar_lambda * m.cvarcost


instance.obj = pyo.Objective(rule=obj_rule)

endTime = time.time() - start
print('%f seconds required to construct' % endTime)


import sys

start = time.time()
instance.write(sys.argv[1])
endTime = time.time() - start
print('%f seconds required to write file %s' % (endTime, sys.argv[1]))

if False:
    for i in instance.SCEN:
        print(
            "Scenario %s = %s"
            % (
                i,
                sum(
                    sum(0.5 * pyo.value(instance.pow[i, j, k]) for j in instance.LINK_A)
                    for k in instance.TIME.get_finite_elements()
                ),
            )
        )

    solver = pyo.SolverFactory('ipopt')
    results = solver.solve(instance, tee=True)

    for i in instance.SCEN:
        print(
            "Scenario %s = %s"
            % (
                i,
                sum(
                    sum(0.5 * pyo.value(instance.pow[i, j, k]) for j in instance.LINK_A)
                    for k in instance.TIME.get_finite_elements()
                ),
            )
        )
