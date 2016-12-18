from pyomo.environ import *

time = range(5)
genset = ['G_MAIN', 'G_EAST']
gennom = {0: 120.0, 1: 145.0, 2: 119.0, 3: 42.0, 4: 190.0}
maxpower = 500
ramplimit = 50

# @usepassedblock:
model = ConcreteModel()
model.TIME = Set(initialize=time, ordered=True)
model.GEN_UNITS = Set(initialize=genset, ordered=True)
def generator_rule(b,g):
    m = b.model()
    b.MaxPower = Param(within=NonNegativeReals, initialize=maxpower)
    b.RampLimit = Param(within=NonNegativeReals, initialize=ramplimit)
    b.Power = Var(m.TIME, bounds=(0,b.MaxPower), initialize=gennom)
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
model.TIME = Set(initialize=time, ordered=True)
model.GEN_UNITS = Set(initialize=genset, ordered=True)

def generator_rule(b,g):
    m = b.model()
    gen = Block(concrete=True)
    gen.MaxPower = Param(within=NonNegativeReals, initialize=maxpower)
    gen.RampLimit = Param(within=NonNegativeReals, initialize=ramplimit)
    gen.Power = Var(m.TIME, bounds=(0,gen.MaxPower), initialize=gennom)
    gen.UnitOn = Var(m.TIME, within=Binary)
    def limit_ramp(_b, t):
        if t == m.TIME.first():
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

# @finaltimepower:
t = model.TIME.last()
for g in model.GEN_UNITS:
    p = model.Generator[g].Power[t]
    print('{0} = {1}'.format(p.name, value(p)))
# @:finaltimepower

# @finaltimepowerslice:
t = model.TIME.last()
for p in model.Generator[:].Power[t]:
    print('{0} = {1}'.format(p.name, value(p)))
# @:finaltimepowerslice
