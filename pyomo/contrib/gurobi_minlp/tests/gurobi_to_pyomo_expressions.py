from pyomo.common.dependencies import attempt_import
from pyomo.core.expr.numeric_expr import (
    SumExpression,
    ProductExpression,
    DivisionExpression,
    PowExpression,
    NegationExpression,
    UnaryFunctionExpression,
    sqrt,
    exp,
    log,
    log10,
    sin,
    cos,
    tan
)

gurobipy, gurobipy_available = attempt_import('gurobipy', minimum_version='12.0.0')

grb_op_to_pyo = {}
if gurobipy_available:
    from gurobipy import GRB

    grb_op_to_pyo.update({
        GRB.OPCODE_PLUS: (SumExpression, ()),
        #GRB.OPCODE_MINUS: , # This is sum of negated term for us
        GRB.OPCODE_UMINUS: (NegationExpression, ()),
        GRB.OPCODE_MULTIPLY: (ProductExpression, ()), # Their multiply is n-ary
        GRB.OPCODE_DIVIDE: (DivisionExpression, ()),
        #GRB.OPCODE_SQUARE: , # This is pow with a fixed second argument for us
        GRB.OPCODE_SQRT: (UnaryFunctionExpression, ('sqrt', sqrt)),
        GRB.OPCODE_EXP: (UnaryFunctionExpression, ('exp', exp)),
        GRB.OPCODE_LOG: (UnaryFunctionExpression, ('log', log)),
        GRB.OPCODE_LOG2: (UnaryFunctionExpression, ('log', log)),
        GRB.OPCODE_LOG10: (UnaryFunctionExpression, ('log10', log10)),
        GRB.OPCODE_POW: (PowExpression, ()),
        GRB.OPCODE_SIN: (UnaryFunctionExpression, ('sin', sin)),
        GRB.OPCODE_COS: (UnaryFunctionExpression, ('cos', cos)),
        GRB.OPCODE_TAN: (UnaryFunctionExpression, ('tan', tan)),
        #GRB.OPCODE_LOGISTIC: We don't have this one.
    })

nary_ops = { SumExpression, }

def grb_nl_to_pyo_expr(op, data, parent, var_map):
    ans = []
    for i, (op, data, parent) in enumerate(zip(op, data, parent)):
        if op in grb_op_to_pyo:
            cls, args = grb_op_to_pyo[op]
            ans.append(cls((), *args))
        elif op == GRB.OPCODE_VARIABLE:
            ans.append(var_map[data])
        elif op == GRB.OPCODE_CONSTANT:
            ans.append(data)
        else:
            raise RuntimeError(
                f"The gurobi-to-pyomo expression converter encountered an unexpected "
                f"(or unsupported) opcode: {op}"
            )
        if i:
            ans[parent]._args_ = ans[parent]._args_ + (ans[-1],)
            if ans[parent].__class__ in nary_ops:
                ans[parent]._nargs += 1

    return ans[0]
