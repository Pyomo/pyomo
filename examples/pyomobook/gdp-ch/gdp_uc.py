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

# gdp_uc.py
import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction

model = pyo.AbstractModel()

# @disjuncts:
model.NumTimePeriods = pyo.Param()
model.GENERATORS = pyo.Set()
model.TIME = pyo.RangeSet(model.NumTimePeriods)

model.MaxPower = pyo.Param(model.GENERATORS, within=pyo.NonNegativeReals)
model.MinPower = pyo.Param(model.GENERATORS, within=pyo.NonNegativeReals)
model.RampUpLimit = pyo.Param(model.GENERATORS, within=pyo.NonNegativeReals)
model.RampDownLimit = pyo.Param(model.GENERATORS, within=pyo.NonNegativeReals)
model.StartUpRampLimit = pyo.Param(model.GENERATORS, within=pyo.NonNegativeReals)
model.ShutDownRampLimit = pyo.Param(model.GENERATORS, within=pyo.NonNegativeReals)


def Power_bound(m, g, t):
    return (0, m.MaxPower[g])


model.Power = pyo.Var(model.GENERATORS, model.TIME, bounds=Power_bound)


def GenOn(b, g, t):
    m = b.model()
    b.power_limit = pyo.Constraint(
        expr=pyo.inequality(m.MinPower[g], m.Power[g, t], m.MaxPower[g])
    )
    if t == m.TIME.first():
        return
    b.ramp_limit = pyo.Constraint(
        expr=pyo.inequality(
            -m.RampDownLimit[g], m.Power[g, t] - m.Power[g, t - 1], m.RampUpLimit[g]
        )
    )


model.GenOn = Disjunct(model.GENERATORS, model.TIME, rule=GenOn)


def GenOff(b, g, t):
    m = b.model()
    b.power_limit = pyo.Constraint(expr=m.Power[g, t] == 0)
    if t == m.TIME.first():
        return
    b.ramp_limit = pyo.Constraint(expr=m.Power[g, t - 1] <= m.ShutDownRampLimit[g])


model.GenOff = Disjunct(model.GENERATORS, model.TIME, rule=GenOff)


def GenStartUp(b, g, t):
    m = b.model()
    b.power_limit = pyo.Constraint(expr=m.Power[g, t] <= m.StartUpRampLimit[g])


model.GenStartup = Disjunct(model.GENERATORS, model.TIME, rule=GenStartUp)
# @:disjuncts


# @disjunction:
def bind_generators(m, g, t):
    return [m.GenOn[g, t], m.GenOff[g, t], m.GenStartup[g, t]]


model.bind_generators = Disjunction(model.GENERATORS, model.TIME, rule=bind_generators)
# @:disjunction


# @logic:
def onState(m, g, t):
    if t == m.TIME.first():
        return pyo.LogicalConstraint.Skip
    return m.GenOn[g, t].indicator_var.implies(
        pyo.lor(m.GenOn[g, t - 1].indicator_var, m.GenStartup[g, t - 1].indicator_var)
    )


model.onState = pyo.LogicalConstraint(model.GENERATORS, model.TIME, rule=onState)


def startupState(m, g, t):
    if t == m.TIME.first():
        return pyo.LogicalConstraint.Skip
    return m.GenStartup[g, t].indicator_var.implies(m.GenOff[g, t - 1].indicator_var)


model.startupState = pyo.LogicalConstraint(
    model.GENERATORS, model.TIME, rule=startupState
)
# @:logic


#
# Fictitious objective to form a legal LP file
#
@model.Objective()
def obj(m):
    return sum(m.Power[g, t] for g in m.GENERATORS for t in m.TIME)


@model.Constraint(model.GENERATORS)
def nontrivial(m, g):
    return sum(m.Power[g, t] for t in m.TIME) >= len(m.TIME) / 2 * m.MinPower[g]


@model.ConstraintList()
def nondegenerate(m):
    for i, g in enumerate(m.GENERATORS):
        yield m.Power[g, i + 1] == 0
