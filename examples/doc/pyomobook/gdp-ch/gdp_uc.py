# gdp_uc.py
from pyomo.environ import *
from pyomo.gdp import *

model = AbstractModel()

# @disjuncts:
model.NumTimePeriods = Param()
model.GENERATORS = Set()
model.TIME = RangeSet(model.NumTimePeriods)

model.MaxPower = Param(model.GENERATORS, within=NonNegativeReals)
model.MinPower = Param(model.GENERATORS, within=NonNegativeReals)
model.RampUpLimit = Param(model.GENERATORS, within=NonNegativeReals)
model.RampDownLimit = Param(model.GENERATORS, within=NonNegativeReals)
model.StartUpRampLimit = Param(model.GENERATORS, within=NonNegativeReals)
model.ShutDownRampLimit = Param(model.GENERATORS, within=NonNegativeReals)

def Power_bound(m,g,t):
    return (0, m.MaxPower[g])
model.Power = Var(model.GENERATORS, model.TIME, bounds=Power_bound)

def GenOn(b, g, t):
    m = b.model()
    b.power_limit = Constraint(
        expr=m.MinPower[g] <= m.Power[g,t] <= m.MaxPower[g] )
    if t == m.TIME.first():
        return
    b.ramp_limit = Constraint(
        expr=-m.RampDownLimit[g] <= m.Power[g,t] - m.Power[g,t-1] <= m.RampUpLimit[g] )
model.GenOn = Disjunct(model.GENERATORS, model.TIME, rule=GenOn)

def GenOff(b, g, t):
    m = b.model()
    b.power_limit = Constraint(
        expr=m.Power[g,t] == 0 )
    if t == m.TIME.first():
        return
    b.ramp_limit = Constraint(
        expr=m.Power[g,t-1] <= m.ShutDownRampLimit[g] )
model.GenOff = Disjunct(model.GENERATORS, model.TIME, rule=GenOff)

def GenStartUp(b, g, t):
    m = b.model()
    b.power_limit = Constraint(
        expr=m.Power[g,t] <= m.StartUpRampLimit[g] )
model.GenStartup = Disjunct(model.GENERATORS, model.TIME, rule=GenStartUp)
# @:disjuncts

# @disjunction:
def bind_generators(m,g,t):
   return [m.GenOn[g,t], m.GenOff[g,t], m.GenStartup[g,t]]
model.bind_generators = Disjunction(model.GENERATORS, model.TIME, rule=bind_generators)
# @:disjunction

# @logic:
def on_state(m,g,t):
    if t == m.TIME.first():
        return Constraint.Skip
    return m.GenOn[g,t].indicator_var <= m.GenOn[g,t-1].indicator_var + m.GenStartup[g,t-1].indicator_var
model.on_state = Constraint(model.GENERATORS, model.TIME, rule=on_state)

def allow_startup(m,g,t):
    if t == m.TIME.first():
        return Constraint.Skip
    return m.GenStartUp[g,t].indicator_var <= m.GenOff[g,t-1].indicator_var
model.allow_startup = Constraint(model.GENERATORS, model.TIME, rule=allow_startup)
# @:logic

