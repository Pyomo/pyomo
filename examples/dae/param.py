from pyomo.environ import *

#Define Model

m = ConcreteModel()

m.eta1 = Param(initialize = 4.5, mutable=True)
m.eta2 = Param(initialize = 1.0, mutable=True)

m.x1 = Var(initialize = 0.15, within=NonNegativeReals)
m.x2 = Var(initialize = 0.15, within=NonNegativeReals)
m.x3 = Var(initialize = 0.0, within=NonNegativeReals)

m.const1 = Constraint(expr=6*m.x1+3*m.x2+2*m.x3-m.eta1 == 0)
m.const2 = Constraint(expr=m.eta2*m.x1+m.x2-m.x3-1 == 0)

m.obj = Objective(expr = m.x1**2+m.x2**2+m.x3**2)

