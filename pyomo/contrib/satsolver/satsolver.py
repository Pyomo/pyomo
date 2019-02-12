
from pyomo.environ import *
from pyomo.core.expr import current as EXPR
import z3
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet)
from pyomo.core.expr.expr_pyomo5 import *

from pyomo.core.expr.expr_pyomo5 import nonpyomo_leaf_types


class SMT_visitor(EXPR.StreamBasedExpressionVisitor):
    """Creates an SMT expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    SMT string

    """

    def __init__(self):
        super(SMT_visitor, self).__init__()

    def exitNode(self,node,data):
        if isinstance(node, EqualityExpression):
            ans = "(= "+data[0] + " " + data[1] + ")"
        elif isinstance(node, InequalityExpression):
            ans = "(<= "+data[0] + " " + data[1] + ")"
        elif isinstance(node, ProductExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = "(* "+ans+" "+ arg +")"
        elif isinstance(node, SumExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = "(+ "+ans+" "+ arg +")"
        elif isinstance(node, PowExpression):
            ans = "(^ "+data[0] + " " + data[1] + ")"
        elif isinstance(node, NegationExpression):
            ans = "(- 0 " + data[0] + ")"
        elif isinstance(node, MonomialTermExpression):
            ans = "(* "+data[0] + " " + data[1] + ")"
        elif isinstance(node, ReciprocalExpression):
            ans = "(/ 1 " + data[0] + ")"
        elif isinstance(node, AbsExpression):
            ans = "(abs "+ data[0] + ")"
        elif isinstance(node, UnaryFunctionExpression):
            if (node.name == "exp"):
                ans = "(exp "+ data[0] + ")"
            elif (node.name == "log"):
                raise NotImplementedError("logarithm not handled by z3 interface")
            elif (node.name == "sin"):
                ans = "(sin "+ data[0] + ")"
            elif (node.name == "cos"):
                ans = "(cos "+ data[0] + ")"
            elif (node.name == "tan"):
                ans = "(tan "+ data[0] + ")"
            elif (node.name == "asin"):
                ans = "(asin "+ data[0] + ")"
            elif (node.name == "acos"):
                ans = "(acos "+ data[0] + ")"
            elif (node.name == "atan"):
                ans = "(atan "+ data[0] + ")"
            elif (node.name == "sqrt"):
                ans = "(^ "+ data[0] + " (/ 1 2))"
            else:
                raise NotImplementedError("Unknown unary function: %s" % (node.name,))
        else:
            raise NotImplementedError(str(type(node)) + " expression not handled by z3 interface")
        return ans
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
def get_default_functions():
    smtstring = ""
    smtstring = smtstring + "(define-fun exp ((x Real)) Real (^ 2.718281828459045 x))"
    return smtstring

#TODO: Get rid of reference to variable names
def isFeasible(model):
    ss = z3.Solver()
    #dict to map between pyomo and z3 vars
    vardict = {}
    smtstring = get_default_functions()
    #add variables to z3 model
    varcount = 0
    for v in model.component_data_objects(ctype = Var, descend_into=True):
        varcount = varcount + 1
        smtstring = addVar(smtstring,v,varcount)
    for c in model.component_data_objects(ctype = Constraint):
        smtstring = addConstraint(smtstring,c.expr)

    ss.append(z3.parse_smt2_string(smtstring))
    return ss.check()

def bound(smtstring,pyomovar):
    nm = pyomovar.name
    lb = pyomovar.lb
    ub = pyomovar.ub
    if lb is not None:
        smtstring = smtstring + "(assert (>= " + nm + " " + str(lb)+"))"
    if ub is not None:
        smtstring = smtstring + "(assert (<= " + nm + " " + str(ub)+"))"
    return smtstring
def addVar(smtstring,pyomovar,i):
    domain = type(pyomovar.domain)
    if domain is RealSet:
        smtstring = smtstring + "(declare-fun "+ pyomovar.name + "() Real)"
        smtstring = bound(smtstring,pyomovar)
    elif domain is IntegerSet:
        smtstring = smtstring + "(declare-fun "+ pyomovar.name + "() Int)"
        smtstring = bound(smtstring,pyomovar)
    elif domain is BooleanSet:
        smtstring = smtstring + "(declare-fun "+ pyomovar.name + "() Bool)"
    return smtstring
def addConstraint(smtstring,expression):
    walker = SMT_visitor()
    return smtstring + "(assert "+walker.walk_expression(expression)+")"


if __name__ == "__main__":
    m = ConcreteModel()
    m.x = Var()
    m.z = Var()
    m.c1 = Constraint(expr= 1 == (m.x))
    m.c2 = Constraint(expr= 2 == (m.x))
    m.o = Objective(expr=m.x*m.z)
    print isFeasible(m)
