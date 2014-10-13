# taxmcp.py	LOR2-AN-MCP-15-14-11 
# Original Pyomo coding by William Hart
# Adapted from AMPL coding by Sven Leyffer, Aug. 2005.

# Optimal tax with two factors of production No other Frills
# Taken from GAMS Model by Miles Light, Department of Economics, 
# University of Colorado, Boulder, 1999.

import pyomo.environ
from pyomo.core import *
from pyomo.mpec import *


model = ConcreteModel()

model.I = RangeSet(1,2)                     # commodities

model.lbar = Param(initialize=2)                        # initial labor endowment	
model.kbar = Param(initialize=2, within=PositiveReals)  # initial capital endowment (assume positive in model)
model.c0 = Param(initialize=3)              # utility index
model.betal = Param(initialize=1)           # leisure consumption
model.rev = Param(initialize=0.5)           # revenues generated
model.sigma = Param(initialize=0.8)         # elasticity of substitution in consumption

model.alpha = Param(model.I, default=0.5)   # labor intensity in production of Y
                                            # try changing the alphas to 0.8 and 0.2
model.phi = Param(model.I, default=1)       # productivity parameter	
model.beta = Param(model.I, default=1)      # consumption shares
  
model.Y = Var(model.I, initialize=1, within=NonNegativeReals)   # Consumption Commodities
model.C = Var(initialize=1, within=NonNegativeReals)            # Utility Production
model.G = Var(within=NonNegativeReals)                          # Government Good Production
model.P = Var(model.I, initialize=1, within=NonNegativeReals)  # Price Of Consumption Commodities
model.PC = Var(initialize=1, within=NonNegativeReals)           # Price Of Utility
model.PL = Var(initialize=1, within=NonNegativeReals)           # Price Of Labor
model.PK = Var(initialize=1, within=NonNegativeReals)           # Price Of Capital
model.PG = Var(initialize=1, within=NonNegativeReals)           # Price Of Government Good
model.GOVT = Var(within=NonNegativeReals)                       # Government Agent
model.T = Var(model.I, within=NonNegativeReals, initialize=0.4) # Endogenous Tax Rate
model.MU = Var(within=NonNegativeReals, initialize=0)           # Shadow Value On Government Constraint 
model.TAU = Var(model.I, initialize=0.5, bounds=(0.4, 0.6))     # Design variables for MPEC ;

# Maximize welfare
model.welfare = Objective(expr=model.C)

# Zero Profit For Activity I
def PROFIT_(model, i):
    return complements(model.PL**model.alpha[i] * model.PK**(1-model.alpha[i]) >= model.phi[i] * model.P[i],
                       model.Y[i] >= 0)
model.PROFIT = Complementarity(model.I, rule=PROFIT_)

# Zero Profit For Welfare Activity
model.PROFITC = Complementarity(expr=complements(
            model.C >= 0,
            (model.betal / model.c0 * model.PL**(1-model.sigma) +
              sum(model.beta[i]/model.c0*(model.P[i]*(1+model.T[i]))**(1-model.sigma) for i in model.I))**(1/(1-model.sigma))
            ))

# Zero Profit For Welfare Government Good
model.PROFITG = Complementarity(expr=complements(model.PG >= model.PL, model.G >= 0))

# Market Clearance For Govt Good
model.MARKETG = Complementarity(expr=complements(model.G*model.PG >= model.GOVT, model.PG >= 0))

# Market Clearance For Commodity I
def MARKET_(model, i):
    return complements(model.Y[i] * model.phi[i] >= (model.PC/(model.P[i]*(1+model.T[i])))**model.sigma * model.beta[i] * model.C,
                       model.P[i] >= 0)
model.MARKET = Complementarity(model.I, rule=MARKET_)

# Market Clearance For Labor
model.MARKETL = Complementarity(expr=complements(
            model.PL * model.lbar >= model.GOVT + sum(model.Y[i] * model.P[i] * model.phi[i] * model.alpha[i] for i in model.I) + model.PL * (model.PC/model.PL)**model.sigma * model.betal * model.C,
            model.PL >= 0))

# Market Clearance For Capital
model.MARKETK = Complementarity(expr=complements(
            model.PK * model.kbar >= sum(model.Y[i] * model.P[i] * model.phi[i] * (1 - model.alpha[i]) for i in model.I),
            model.PK >= 0))

# Government Income Balance
model.REVENUE = Complementarity(expr=complements(
            model.GOVT >= sum(model.Y[i] * model.phi[i] * model.P[i] * model.T[i] for i in model.I),
            model.GOVT >= 0))

# Household Income Balance
model.INCOME = Complementarity(expr=complements(
            model.PC * model.C * model.c0 >= model.PL * model.lbar + model.PK * model.kbar,
            model.PC >= 0))

# Revenue Constraint
model.REV_CONST = Complementarity(expr=complements(
            model.GOVT >= model.PL * model.rev,
            model.MU >= 0))

# Tax Definition  
def TAX_(model, i):
    return complements(model.T[i] >= model.MU * model.TAU[i], model.T[i] >= 0)
model.TAX = Complementarity(model.I, rule=TAX_)

# numeraire
model.PL.fix()

