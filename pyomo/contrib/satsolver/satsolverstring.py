from pyomo.environ import *
import z3
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet)
from pyomo.core.expr.expr_pyomo5 import (
    AbsExpression, LinearExpression, NegationExpression, NPV_AbsExpression,
    NPV_ExternalFunctionExpression, NPV_NegationExpression, NPV_PowExpression,
    NPV_ProductExpression, NPV_ReciprocalExpression, NPV_SumExpression,
    NPV_UnaryFunctionExpression, PowExpression, ProductExpression,
    ReciprocalExpression, StreamBasedExpressionVisitor, SumExpression,
    UnaryFunctionExpression, nonpyomo_leaf_types
)
def isFeasible(model):
    ss = z3.Solver()
    #dict to map between pyomo and z3 vars
    vardict = {}
    smtstring = ""
    #add variables to z3 model
    for v in model.component_data_objects(ctype = Var, descend_into=True):
        smtstring = addVar(smtstring,v,vardict)
    print smtstring
    smtstring = smtstring + "(assert (= x 3))"
    print z3.parse_smt2_string(smtstring)
    ss.append(z3.parse_smt2_string(smtstring))
    return ss.check() == z3.sat

def bound(smtstring,pyomovar):
    nm = pyomovar.name
    lb = pyomovar.lb
    ub = pyomovar.ub
    smtstring = smtstring + "(assert (or (= " + nm + " " + str(lb) + ") (> " + nm + " " + str(lb)+")))"
    return smtstring + "(assert (or (= " + nm + " " + str(ub) + ") (< " + nm + " " + str(ub)+")))"
def addVar(smtstring,pyomovar,vardict):
    domain = type(pyomovar.domain)
    if domain is RealSet:
        print "Real " + pyomovar.name
        smtstring = smtstring + "(declare-fun "+ pyomovar.name + "() Real)"
        smtstring = bound(smtstring,pyomovar)
    elif domain is IntegerSet:
        print "Int " + pyomovar.name
        smtstring = smtstring + "(declare-fun "+ pyomovar.name + "() Int)"
        smtstring = bound(smtstring,pyomovar)
    elif domain is BooleanSet:
        print "Bool " + pyomovar.name
        smtstring = smtstring + "(declare-fun "+ pyomovar.name + "() Bool)"
    return smtstring



if __name__ == "__main__":
    m = ConcreteModel()
    m.x = Var(domain = NonNegativeReals,bounds = (0,3))
    m.y = Var(domain = Boolean)
    m.z = Var(domain = PositiveIntegers, bounds = (11,20))
    m.c = Constraint(expr=m.x == m.z + 3)
    m.o = Objective(expr=m.x*m.y)
    print isFeasible(m)
