from pyomo.environ import *
from pyomo.mpec import *
from pyomo.dae import *
from pyomo.gdp import *
from pyomo.bilevel import *

model = ConcreteModel()

# Set
model.A = Set(initialize=[1,2])
model.B = Set(initialize=[3,4])
model.C = Set(model.A, model.B)

# Block
model.b1 = Block()
model.b2 = Block(model.A, model.B)

# BuildAction
def rule(model):
    return True
model.ba1 = BuildAction(rule=rule)
def rule(model, i, j):
    return True
model.ba2 = BuildAction(model.A, model.B, rule=rule)

# BuildCheck
def rule(model):
    return True
model.bc1 = BuildCheck(rule=rule)
def rule(model, i, j):
    return True
model.bc2 = BuildCheck(model.A, model.B, rule=rule)

# Complementarity
model.comp1 = Complementarity()
model.comp2 = Complementarity(model.A, model.B)

# ComplementarityList
model.complist = ComplementarityList()

# Constraint
model.c1 = Constraint()
model.c2 = Constraint(model.A, model.B)

# ConstraintList
model.clist1 = ConstraintList()

# ContinuousSet
#model.cset1 = ContinuousSet()
#model.cset2 = ContinuousSet(model.A, model.B)

# DerivativeVar
#model.dvar1 = DerivativeVar()
#model.dvar2 = DerivativeVar(model.A, model.B)

# Disjunct
model.disjunct1 = Disjunct()
model.disjunct2 = Disjunct(model.A, model.B)

# Disjunction
#model.disjunction1 = Disjunction()
#model.disjunction2 = Disjunction(model.A, model.B)

# Expression
model.expr1 = Expression()
model.expr2 = Expression(model.A, model.B)

# Integral
#model.int1 = Integral()
#model.int2 = Integral(model.A, model.B)

# Objective
model.o1 = Objective()
model.o2 = Objective(model.A, model.B)

# ObjectiveList
model.olist1 = ObjectiveList()

# Param
model.a = Param(initialize=1.1)
model.b = Param(model.A, model.B)

# Piecewise
# TODO: fix Piecewise to be a true indexed component
#model.pwise1 = Piecewise()
#model.pwise2 = Piecewise(model.A, model.B)

# RangeSet
#model.rset1 = RangeSet()
#model.rset2 = RangeSet(model.A, model.B)

# SubModel
model.sub1 = SubModel()
model.sub2 = SubModel(model.A, model.B)

# Suffix
#model.suf1 = Suffix()
#model.suf2 = Suffix(model.A, model.B)

# Var
model.x = Var()
model.y = Var(model.A, model.B, initialize=True)
model.z = Var(model.A, model.B, initialize=True)

# VarList
model.vlist1 = VarList()

# SOSConstraint
model.sset1 = SOSConstraint(var=model.y, sos=1)
model.sset2 = SOSConstraint(model.A, model.B, var=model.z, sos=1)

model.pprint()
