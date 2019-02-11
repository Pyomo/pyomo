from pyomo.environ import *
import z3
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet)

def isFeasible(model):
    ss = z3.Solver()
    #dict to map between pyomo and z3 vars
    vardict = {}
    #add variables to z3 model
    for v in model.component_data_objects(ctype = Var, descend_into=True):
        addVar(ss,v,vardict)
    return ss.check() == z3.sat

def bound(satsolver,z3var,lb,ub):
    satsolver.add(z3var >= lb)
    satsolver.add(z3var <= ub)

def addVar(satsolver,pyomovar,vardict):
    domain = type(pyomovar.domain)
    if domain is RealSet:
        print "Real " + pyomovar.name
        z3var = z3.Real(pyomovar.name)
        bound(satsolver,z3var,pyomovar.lb,pyomovar.ub)
    elif domain is IntegerSet:
        print "Int " + pyomovar.name
        z3var = z3.Int(pyomovar.name)
        bound(satsolver,z3var,pyomovar.lb,pyomovar.ub)
    elif domain is BooleanSet:
        print "Bool " + pyomovar.name
        z3var = z3.Int(pyomovar.name)
    vardict[pyomovar.name] = z3var









if __name__ == "__main__":
    m = ConcreteModel()
    m.x = Var(domain = NonNegativeReals,bounds = (0,10))
    m.y = Var(domain = Boolean)
    m.z = Var(domain = PositiveIntegers, bounds = (11,20))
    m.c = Constraint(expr=m.x == m.y + 3)
    m.o = Objective(expr=m.x*m.y)
    print isFeasible(m)
