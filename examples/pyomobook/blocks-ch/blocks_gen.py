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

import pyomo.environ as pyo

time = range(5)
genset = ['G_MAIN', 'G_EAST']
gennom = {0: 120.0, 1: 145.0, 2: 119.0, 3: 42.0, 4: 190.0}
maxpower = 500
ramplimit = 50

# @usepassedblock:
model = pyo.ConcreteModel()
model.TIME = pyo.Set(initialize=time, ordered=True)
model.GEN_UNITS = pyo.Set(initialize=genset, ordered=True)


def generator_rule(b, g):
    m = b.model()
    b.MaxPower = pyo.Param(within=pyo.NonNegativeReals, initialize=maxpower)
    b.RampLimit = pyo.Param(within=pyo.NonNegativeReals, initialize=ramplimit)
    b.Power = pyo.Var(m.TIME, bounds=(0, b.MaxPower), initialize=gennom)
    b.UnitOn = pyo.Var(m.TIME, within=pyo.Binary)

    def limit_ramp(_b, t):
        if t == min(_b.model().TIME):
            return pyo.Constraint.Skip
        return pyo.inequality(
            -_b.RampLimit, _b.Power[t] - _b.Power[t - 1], _b.RampLimit
        )

    b.limit_ramp = pyo.Constraint(m.TIME, rule=limit_ramp)
    b.CostCoef = pyo.Param([1, 2])

    def Cost(_b, t):
        return sum(_b.CostCoef[i] * _b.Power[t] ** i for i in _b.CostCoef)

    b.Cost = pyo.Expression(m.TIME, rule=Cost)


model.Generator = pyo.Block(model.GEN_UNITS, rule=generator_rule)
# @:usepassedblock

model.pprint()
model = None

# @buildnewblock:
model = pyo.ConcreteModel()
model.TIME = pyo.Set(initialize=time, ordered=True)
model.GEN_UNITS = pyo.Set(initialize=genset, ordered=True)


def generator_rule(b, g):
    m = b.model()
    gen = pyo.Block(concrete=True)
    gen.MaxPower = pyo.Param(within=pyo.NonNegativeReals, initialize=maxpower)
    gen.RampLimit = pyo.Param(within=pyo.NonNegativeReals, initialize=ramplimit)
    gen.Power = pyo.Var(m.TIME, bounds=(0, gen.MaxPower), initialize=gennom)
    gen.UnitOn = pyo.Var(m.TIME, within=pyo.Binary)

    def limit_ramp(_b, t):
        if t == m.TIME.first():
            return pyo.Constraint.Skip
        return pyo.inequality(
            -_b.RampLimit, _b.Power[t] - _b.Power[t - 1], _b.RampLimit
        )

    gen.limit_ramp = pyo.Constraint(m.TIME, rule=limit_ramp)
    gen.CostCoef = pyo.Param([1, 2])

    def Cost(_b, t):
        return sum(_b.CostCoef[i] * _b.Power[t] ** i for i in _b.CostCoef)

    gen.Cost = pyo.Expression(m.TIME, rule=Cost)
    return gen


model.Generator = pyo.Block(model.GEN_UNITS, rule=generator_rule)
# @:buildnewblock

model.pprint()

# @finaltimepower:
t = model.TIME.last()
for g in model.GEN_UNITS:
    p = model.Generator[g].Power[t]
    print('{0} = {1}'.format(p.name, pyo.value(p)))
# @:finaltimepower

# @finaltimepowerslice:
t = model.TIME.last()
for p in model.Generator[:].Power[t]:
    print('{0} = {1}'.format(p.name, pyo.value(p)))
# @:finaltimepowerslice
