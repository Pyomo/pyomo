from pyomo.environ import *
from pyomo.core.expr import current as EXPR
import z3
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet)
from pyomo.core.expr.expr_pyomo5 import *

from pyomo.core.expr.expr_pyomo5 import nonpyomo_leaf_types


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




class SMT_visitor(EXPR.StreamBasedExpressionVisitor):
    """Creates an SMT expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    SMT string

    """

    def __init__(self):
        super(SMT_visitor, self).__init__()

    def exitNode(self,node,data):
        print "new node:"
        print node
        if isinstance(node, EqualityExpression):
            print('=')
            return "(= "+data[0] + " " + data[1] + ")"
        elif isinstance(node, SumExpression):
            print('+')
            return "(+ "+data[0] + " " + data[1] + ")"
        elif isinstance(node, NegationExpression):
            print('Neg')
            return "(- 0 " + data[0] + ")"
        else:
            print node
    def beforeChild(self,node,child):
        if type(child) in nonpyomo_leaf_types:
            # This means the child is POD
            # i.e., int, float, string
            return False,str(child)
        elif not child.is_expression_type():
            # node is either a Param, Var, or NumericConstant
            return False,str(child)
        else:
            # this is an expression node
            return True,""

    def finalizeResult(self, node_result):
        return node_result





if __name__ == "__main__":
    m = ConcreteModel()
    m.x = Var(domain = NonNegativeReals,bounds = (0,10))
    m.y = Var(domain = Boolean)
    m.z = Var(domain = PositiveIntegers, bounds = (11,20))
    m.c = Constraint(expr=m.x == m.y + 3)
    m.o = Objective(expr=m.x*m.y)
    walker  = SMT_visitor()
    print walker.walk_expression(m.c.expr)
    print isFeasible(m)
