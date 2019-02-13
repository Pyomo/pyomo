

from pyomo.core.expr import current as EXPR
import z3
from pyomo.core import value, Expression
from pyomo.core.kernel.set_types import (RealSet,
                                         IntegerSet,
                                         BooleanSet)
from pyomo.core.expr.expr_pyomo5 import (EqualityExpression,
                                        InequalityExpression,
                                        ProductExpression,
                                        SumExpression,
                                        PowExpression,
                                        NegationExpression,
                                        MonomialTermExpression,
                                        ReciprocalExpression,
                                        AbsExpression,
                                        UnaryFunctionExpression,
                                        nonpyomo_leaf_types
                                        )
from pyomo.environ import SymbolMap,NumericLabeler

class SMTSatSolver(object):
    def __str__(self):
        string = ""
        string = string + "Variables:\n"
        for v in self.variable_list:
            string = string + v + "\n"
        string = string + "Bounds:\n"
        for e in self.bounds_list:
            string = string + v + "\n"
        string = string + "Expressions:\n"
        for e in self.expression_list:
            string = string + v + "\n"
    def __init__(self,model = None):
        self.variable_label_map = SymbolMap(NumericLabeler('x'))
        self.prefix_expr_list = []
        self.variable_list = []
        self.bounds_list = []
        self.expression_list= []
        self.walker = SMT_visitor(self.variable_label_map)
        if model is not None:
            self._process_model(model)
    def _get_default_functions(self):
        self.prefix_expr_list.append("(define-fun exp ((x Real)) Real (^ 2.718281828459045 x))")
    def _process_model(self,model):
        for v in model.component_data_objects(ctype = Var, descend_into=True):
            smtstring = self.add_var(v)
        for c in model.component_data_objects(ctype = Constraint):
            smtstring = self.add_expr(c.expr)
    def _add_bound(self,var):
        nm = self.variable_label_map.getSymbol(var)
        lb = var.lb
        ub = var.ub
        if lb is not None:
            self.bounds_list.append ("(assert (>= " + nm + " " + str(lb)+"))")
        if ub is not None:
            self.bounds_list.append ("(assert (<= " + nm + " " + str(ub)+"))")
    def add_var(self,var):
        label = self.variable_label_map.getSymbol(var)
        domain = type(var.domain)
        if domain is RealSet:
            self.variable_list.append("(declare-fun "+ label + "() Real)")
            self._add_bound(var)
        elif domain is IntegerSet:
            self.variable_list.append("(declare-fun "+ label + "() Int)")
            self._add_bound(var)
        elif domain is BooleanSet:
            self.variable_list.append("(declare-fun "+ label + "() Bool)")
        else:
            raise NotImplementedError("SMT cannot handle" + str(domain) + "variables")
    def add_expr(self,expression):
        smtexpr = self.walker.walk_expression(expression)
        self.expression_list.append ("(assert " +smtexpr+")")
    def check(self):
        prefix_string = ''.join(self.prefix_expr_list)
        variable_string = ''.join(self.variable_list)
        bounds_string = ''.join(self.bounds_list)
        expression_string = ''.join(self.expression_list)
        smtstring = prefix_string + variable_string +bounds_string + expression_string
        ss = z3.Solver()
        ss.append(z3.parse_smt2_string(smtstring))
        return ss.check()


class SMT_visitor(EXPR.StreamBasedExpressionVisitor):
    """Creates an SMT expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    SMT string

    """

    def __init__(self,varmap):
        super(SMT_visitor, self).__init__()
        self.variable_label_map = varmap
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
        elif child.is_variable_type():
            return False,str(self.variable_label_map.getSymbol(child))
        elif child.is_parameter_type():
            return False,str(value(child))
        elif not child.is_expression_type():
            return False,str(child)
        else:
            # this is an expression node
            return True,""
    def finalizeResult(self, node_result):
        return node_result
