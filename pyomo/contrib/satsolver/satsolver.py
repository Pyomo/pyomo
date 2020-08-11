import math 

from pyomo.common.dependencies import attempt_import
from pyomo.core import value, SymbolMap, NumericLabeler, Var, Constraint
from pyomo.core.expr.logical_expr import (
    EqualityExpression,
    InequalityExpression,
)
from pyomo.core.expr.numeric_expr import (
    ProductExpression,
    SumExpression,
    PowExpression,
    NegationExpression,
    MonomialTermExpression,
    DivisionExpression,
    ReciprocalExpression,
    AbsExpression,
    UnaryFunctionExpression,
)
from pyomo.core.expr.numvalue import (
    nonpyomo_leaf_types,
)
from pyomo.core.expr.visitor import (
    StreamBasedExpressionVisitor,
)
from pyomo.gdp import Disjunction

z3, z3_available = attempt_import('z3')


def satisfiable(model, logger=None):
    """Checks if the model is satisfiable.

    Returns:
        True  if model is satisfiable,
        False if model is unsatisfiable,
        None  if satisfiability cannot be determined.
    """
    result = SMTSatSolver(model, logger=logger).check()
    if result == z3.sat:
        return True
    elif result == z3.unsat:
        return False
    elif result == z3.unknown:
        return None
    else:
        raise ValueError('Unknown result: %s' % result)


class SMTSatSolver(object):
    """
    Satisfiability solver that checks constraint feasibility through use of
    z3 Sat Solver. Object stores expressions and variables in form consistent
    with SMT-LIB standard.
    For documentation on SMT-LIB standard see
    http://smtlib.cs.uiowa.edu/
    """

    def __str__(self):
        """
        Defined string representation of object
        """
        string = ""
        string = string + "Variables:\n"
        for v in self.variable_list:
            string = string + v
        string = string + "Bounds:\n"
        for e in self.bounds_list:
            string = string + e
        string = string + "Expressions:\n"
        for e in self.expression_list:
            string = string + e
        string = string + "Disjunctions:\n"
        for djn in self.disjunctions_list:
            string = string + "Disjunction: " + djn[0] + "\n"
            for disj in djn[1]:
                string = string + "  " + disj[0] + " : " + "\n"
                for c in disj[1]:
                    string = string + "    " + c + "\n"
        return string

    def __init__(self, model=None, logger=None):
        self.variable_label_map = SymbolMap(NumericLabeler('x'))
        self.prefix_expr_list = self._get_default_functions()
        self.variable_list = []
        self.bounds_list = []
        self.expression_list = []
        self.disjunctions_list = []
        self.walker = SMT_visitor(self.variable_label_map)
        self.solver = z3.Solver()
        self.logger = logger
        if model is not None:
            self._process_model(model)

    # Set up functions to be added to beginning of string
    def _get_default_functions(self):
        default = list()
        default.append("(define-fun exp ((x Real)) Real (^ %0.15f x))" % (math.exp(1),))
        return default

    # processes pyomo model into SMT model
    def _process_model(self, model):
        for v in model.component_data_objects(ctype=Var, descend_into=True):
            smtstring = self.add_var(v)
        for c in model.component_data_objects(ctype=Constraint, active=True):
            self.add_expr(c.expr)
        for djn in model.component_data_objects(ctype=Disjunction):
            if djn.active:
                self._process_active_disjunction(djn)
            else:
                self._process_inactive_disjunction(djn)

    # define bound constraints
    def _add_bound(self, var):
        nm = self.variable_label_map.getSymbol(var)
        lb = var.lb
        ub = var.ub
        if lb is not None:
            self.bounds_list.append("(assert (>= " + nm + " " + str(lb) + "))\n")
        if ub is not None:
            self.bounds_list.append("(assert (<= " + nm + " " + str(ub) + "))\n")

    # define variables
    def add_var(self, var):
        label = self.variable_label_map.getSymbol(var)
        domain = var.domain
        if var.is_continuous():
            self.variable_list.append("(declare-fun " + label + "() Real)\n")
            self._add_bound(var)
        elif var.is_binary():
            self.variable_list.append("(declare-fun " + label + "() Int)\n")
            self._add_bound(var)
        elif var.is_integer():
            self.variable_list.append("(declare-fun " + label + "() Int)\n")
            self._add_bound(var)
        else:
            raise NotImplementedError("SMT cannot handle " + str(domain) + " variables")
        return label

    # Defines SMT expression from pyomo expression
    def add_expr(self, expression):
        try:
            smtexpr = self.walker.walk_expression(expression)
            self.expression_list.append("(assert " + smtexpr + ")\n")
        except NotImplementedError as e:
            if self.logger is not None:
                self.logger.warning("Skipping Expression: " + str(e))

    # Computes the SMT Model for the disjunction from the internal class storage
    def _compute_disjunction_string(self, smt_djn):
        djn_string = smt_djn[0]
        for disj in smt_djn[1]:
            cons_string = "true"
            for c in disj[1]:
                cons_string = "(and " + cons_string + " " + c + ")"
            djn_string = djn_string + "(assert (=> ( = 1 " + disj[0] + ") " + cons_string + "))\n"
        return djn_string

    # converts disjunction to internal class storage
    def _process_active_disjunction(self, djn):
        or_expr = "0"
        disjuncts = []
        for disj in djn.disjuncts:
            constraints = []
            iv = disj.indicator_var
            label = self.add_var(iv)
            or_expr = "(+ " + or_expr + " " + label + ")"
            for c in disj.component_data_objects(ctype=Constraint, active=True):
                try:
                    constraints.append(self.walker.walk_expression(c.expr))
                except NotImplementedError as e:
                    if self.logger is not None:
                        self.logger.warning("Skipping Disjunct Expression: " + str(e))
            disjuncts.append((label, constraints))
        if djn.xor:
            or_expr = "(assert (= 1 " + or_expr + "))\n"
        else:
            or_expr = "(assert (>= 1 " + or_expr + "))\n"
        self.disjunctions_list.append((or_expr, disjuncts))

    # processes inactive disjunction indicator vars without constraints
    def _process_inactive_disjunction(self, djn):
        or_expr = "0"
        for disj in djn.disjuncts:
            iv = disj.indicator_var
            label = self.add_var(iv)
            or_expr = "(+ " + or_expr + " " + label + ")"
        if djn.xor:
            or_expr = "(assert (= 1 " + or_expr + "))\n"
        else:
            or_expr = "(assert (>= 1 " + or_expr + "))\n"
        self.expression_list.append(or_expr)

    def get_SMT_string(self):
        prefix_string = ''.join(self.prefix_expr_list)
        variable_string = ''.join(self.variable_list)
        bounds_string = ''.join(self.bounds_list)
        expression_string = ''.join(self.expression_list)
        disjunctions_string = ''.join([self._compute_disjunction_string(d) for d in self.disjunctions_list])
        smtstring = prefix_string + variable_string + bounds_string + expression_string + disjunctions_string
        return smtstring

    def get_var_dict(self):
        labels = [x for x in self.variable_label_map.bySymbol]
        labels.sort()
        vars = [self.variable_label_map.getObject(l) for l in labels]
        return zip(labels, vars)

    # Checks Satisfiability of model
    def check(self):
        self.solver.append(z3.parse_smt2_string(self.get_SMT_string()))
        return self.solver.check()


class SMT_visitor(StreamBasedExpressionVisitor):
    """Creates an SMT expression from the corresponding Pyomo expression.

    This class walks a pyomo expression tree and builds up the corresponding
    SMT string representation of an equivalent expression

    """

    def __init__(self, varmap):
        super(SMT_visitor, self).__init__()
        self.variable_label_map = varmap

    def exitNode(self, node, data):
        if isinstance(node, EqualityExpression):
            ans = "(= " + data[0] + " " + data[1] + ")"
        elif isinstance(node, InequalityExpression):
            ans = "(<= " + data[0] + " " + data[1] + ")"
        elif isinstance(node, ProductExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = "(* " + ans + " " + arg + ")"
        elif isinstance(node, SumExpression):
            ans = data[0]
            for arg in data[1:]:
                ans = "(+ " + ans + " " + arg + ")"
        elif isinstance(node, PowExpression):
            ans = "(^ " + data[0] + " " + data[1] + ")"
        elif isinstance(node, NegationExpression):
            ans = "(- 0 " + data[0] + ")"
        elif isinstance(node, MonomialTermExpression):
            ans = "(* " + data[0] + " " + data[1] + ")"
        elif isinstance(node, DivisionExpression):
            ans = "(/ " + data[0] + " " + data[1] + ")"
        elif isinstance(node, ReciprocalExpression):
            ans = "(/ 1 " + data[0] + ")"
        elif isinstance(node, AbsExpression):
            ans = "(abs " + data[0] + ")"
        elif isinstance(node, UnaryFunctionExpression):
            if node.name == "exp":
                ans = "(exp " + data[0] + ")"
            elif node.name == "log":
                raise NotImplementedError("logarithm not handled by z3 interface")
            elif node.name == "sin":
                ans = "(sin " + data[0] + ")"
            elif node.name == "cos":
                ans = "(cos " + data[0] + ")"
            elif node.name == "tan":
                ans = "(tan " + data[0] + ")"
            elif node.name == "asin":
                ans = "(asin " + data[0] + ")"
            elif node.name == "acos":
                ans = "(acos " + data[0] + ")"
            elif node.name == "atan":
                ans = "(atan " + data[0] + ")"
            elif node.name == "sqrt":
                ans = "(^ " + data[0] + " (/ 1 2))"
            else:
                raise NotImplementedError("Unknown unary function: %s" % (node.name,))
        else:
            raise NotImplementedError(str(type(node)) + " expression not handled by z3 interface")
        return ans

    def beforeChild(self, node, child, child_idx):
        if type(child) in nonpyomo_leaf_types:
            # This means the child is POD
            # i.e., int, float, string
            return False, str(child)
        elif child.is_expression_type():
            return True, ""
        elif child.is_numeric_type():
            if child.is_fixed():
                return False, str(value(child))
            else:
                return False, str(self.variable_label_map.getSymbol(child))
        else:
            return False, str(child)

    def finalizeResult(self, node_result):
        return node_result
