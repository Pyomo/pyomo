from pyomo.environ import *

# @usepassedblock:
model = ConcreteModel()
model.TIME = Set()
model.GEN_UNITS = Set()
def generator_rule(b,g):
    m = b.model()
    b.MaxPower = Param(within=NonNegativeReals)
    b.RampLimit = Param(within=NonNegativeReals)
    b.Power = Var(m.TIME, bounds=(0,b.MaxPower))
    b.UnitOn = Var(m.TIME, within=Binary)
    def limit_ramp(_b, t):
        if t == min(_b.model().TIME):
            return Constraint.Skip
        return -_b.RampLimit <= _b.Power[t] - _b.Power[t-1] <= _b.RampLimit
    b.limit_ramp = Constraint(m.TIME, rule=limit_ramp)
    b.CostCoef = Param([1,2])
    def Cost(_b,t):
        return sum(_b.CostCoef[i]*_b.Power[t]**i for i in _b.CostCoef)
    b.Cost = Expression(m.TIME, rule=Cost)
    
model.Generator = Block(model.GEN_UNITS, rule=generator_rule)
# @:usepassedblock

model.pprint()
model = None

# @buildnewblock:
model = ConcreteModel()
model.TIME = Set()
model.GEN_UNITS = Set()
def generator_rule(b,g):
    m = b.model()
    gen = Block(concrete=True)
    gen.MaxPower = Param(within=NonNegativeReals)
    gen.RampLimit = Param(within=NonNegativeReals)
    gen.Power = Var(m.TIME, bounds=(0,gen.MaxPower))
    gen.UnitOn = Var(m.TIME, within=Binary)
    def limit_ramp(_b, t):
        if t == _b.model().TIME.first():
            return Constraint.Skip
        return -_b.RampLimit <= _b.Power[t] - _b.Power[t-1] <= _b.RampLimit
    gen.limit_ramp = Constraint(m.TIME, rule=limit_ramp)
    gen.CostCoef = Param([1,2])
    def Cost(_b,t):
        return sum(_b.CostCoef[i]*_b.Power[t]**i for i in _b.CostCoef)
    gen.Cost = Expression(m.TIME, rule=Cost)
    return gen

model.Generator = Block(model.GEN_UNITS, rule=generator_rule)
# @:buildnewblock

model.pprint()
