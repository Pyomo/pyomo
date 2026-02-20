from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.common.numeric_types import native_numeric_types
from pyomo.core.expr.numeric_expr import (
    NegationExpression,
    PowExpression,
    ProductExpression,
    MonomialTermExpression,
    DivisionExpression,
    SumExpression,
    LinearExpression,
    UnaryFunctionExpression,
)

from pyomo.repn.util import ExitNodeDispatcher
from pyomo.core.base import (
    VarData, 
    ParamData, 
    ExpressionData, 
    VarList, 
    ConstraintList, 
    Block,
    Constraint,
    Objective,
)
from pyomo.core.base.var import ScalarVar
from pyomo.core.base.param import ScalarParam
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr, fbbt
from pyomo.core.base.expression import ScalarExpression
from pyomo.core.base.transformation import Transformation, TransformationFactory
from pyomo.common.modeling import unique_component_name
from pyomo.core.base.component import ActiveComponent
from pyomo.core.base.suffix import Suffix


def _handle_var(node, data, visitor):
    if node.fixed:
        return _handle_float(node.value, data, visitor)
    visitor.node_to_var_map[node] = (node,)
    visitor.degree_map[node] = 1
    visitor.substitution_map[node] = node
    return node


def _handle_param(node, data, visitor):
    return _handle_float(node.value, data, visitor)


def _handle_float(node, data, visitor):
    visitor.node_to_var_map[node] = tuple()
    visitor.degree_map[node] = 0
    visitor.substitution_map[node] = node
    return node


def _handle_product(node, data, visitor):
    arg1, arg2 = data
    arg1_vars = visitor.node_to_var_map[arg1]
    arg2_vars = visitor.node_to_var_map[arg2]
    arg1_nvars = len(arg1_vars)
    arg2_nvars = len(arg2_vars)
    arg1_degree = visitor.degree_map[arg1]
    arg2_degree = visitor.degree_map[arg2]

    if arg1_degree == 0:
        res = arg1 * arg2
        visitor.node_to_var_map[res] = arg2_vars
        visitor.degree_map[res] = arg2_degree
        visitor.substitution_map[node] = res
        return res
    
    if arg2_degree == 0:
        res = arg1 * arg2
        visitor.node_to_var_map[res] = arg1_vars
        visitor.degree_map[res] = arg1_degree
        visitor.substitution_map[node] = res
        return res

    if arg1_nvars > 1:
        arg1 = visitor.create_aux_var(arg1)
        arg1_vars = (arg1,)
        arg1_nvars = 1
        arg1_degree = 1
    if arg2_nvars > 1:
        arg2 = visitor.create_aux_var(arg2)
        arg2_vars = (arg2,)
        arg2_nvars = 1
        arg2_degree = 1
    res = arg1 * arg2
    # at this point arg1 should have at most 1 variable 
    # and arg2 should have at most 1 variable
    if arg1_nvars == 0:
        visitor.node_to_var_map[res] = arg2_vars
    elif arg2_nvars == 0:
        visitor.node_to_var_map[res] = arg1_vars
    else:
        x = arg1_vars[0]
        y = arg2_vars[0]
        if x is y:
            visitor.node_to_var_map[res] = (x,)
        else:
            visitor.node_to_var_map[res] = (x, y)
    if arg1_degree == 0:
        visitor.degree_map[res] = arg2_degree
    elif arg2_degree == 0:
        visitor.degree_map[res] = arg1_degree
    else:
        visitor.degree_map[res] = -1
    visitor.substitution_map[node] = res
    return res


def _handle_sum(node, data, visitor):
    arg_list = []
    new_degree = 0
    vset = ComponentSet()
    for arg in data:
        arg_vars = visitor.node_to_var_map[arg]
        arg_degree = visitor.degree_map[arg]
        if arg_degree == -1:
            arg = visitor.create_aux_var(arg)
            arg_vars = (arg,)
            arg_degree = 1
        arg_list.append(arg)
        if arg_degree != 0:
            new_degree = 1
        vset.update(arg_vars)
    res = sum(arg_list)
    visitor.node_to_var_map[res] = tuple(vset)
    visitor.degree_map[res] = new_degree
    visitor.substitution_map[node] = res
    return res


def _handle_division(node, data, visitor):
    """
    This one is a bit tricky. If we encounter both x/z and y/z 
    at different places in the model, we only want one auxilliary 
    variable for 1/z.
    """
    arg1, arg2 = data
    arg1_vars = visitor.node_to_var_map[arg1]
    arg2_vars = visitor.node_to_var_map[arg2]
    arg1_nvars = len(arg1_vars)
    arg2_nvars = len(arg2_vars)
    arg1_degree = visitor.degree_map[arg1]
    arg2_degree = visitor.degree_map[arg2]

    if arg2_degree == 0:
        res = arg1 / arg2
        visitor.node_to_var_map[res] = arg1_vars
        visitor.degree_map[res] = arg1_degree
        visitor.substitution_map[node] = res
        return res

    if arg1_nvars > 1:
        arg1 = visitor.create_aux_var(arg1)
        arg1_vars = (arg1,)
        arg1_nvars = 1
        arg1_degree = 1

    if arg2_nvars > 1:
        arg2 = visitor.create_aux_var(arg2)
        arg2_vars = (arg2,)
        arg2_nvars = 1
        arg2_degree = 1

    if "div" not in visitor.substitution_map:
        visitor.substitution_map["div"] = ComponentMap()

    # now we need to figure out if we have seen 1/arg2 before
    if arg2 in visitor.substitution_map["div"]:
        aux = visitor.substitution_map["div"][arg2]
    else:
        aux = visitor.block.x.add()
        visitor.substitution_map["div"][arg2] = aux
        """
        we can only create a piecewise linear function of 1/arg2 if arg2 is either
        strictly greater than 0 or strictly less than 0

        otherwise, we do
        aux = 1 / arg2
        aux * arg2 = 1
        """
        arg2_lb, arg2_ub = compute_bounds_on_expr(arg2)
        if (arg2_lb is not None and arg2_lb > 0) or (arg2_ub is not None and arg2_ub < 0):
            c = visitor.block.c.add(aux == 1/arg2)  # keep it univariate if we can
        else:
            c = visitor.block.c.add(aux * arg2 == 1)
        fbbt(c)

    arg2 = aux
    arg2_vars = (arg2,)
    arg2_nvars = 1
    arg2_degree = 1
    res = arg1 * arg2
    # at this point arg1 should have at most 1 variable 
    # and arg2 should have exactly 1 variable
    if arg1_nvars == 0:
        visitor.node_to_var_map[res] = arg2_vars
    else:
        x = arg1_vars[0]
        y = arg2_vars[0]
        if x is y:
            visitor.node_to_var_map[res] = (x,)
        else:
            visitor.node_to_var_map[res] = (x, y)
    if arg1_degree == 0:
        visitor.degree_map[res] = arg2_degree
    else:
        visitor.degree_map[res] = -1
    visitor.substitution_map[node] = res
    return res


def _handle_pow(node, data, visitor):
    # arg1 ** arg2
    # exp(arg2 * log(arg1))
    arg1, arg2 = data
    arg1_vars = visitor.node_to_var_map[arg1]
    arg2_vars = visitor.node_to_var_map[arg2]
    arg1_nvars = len(arg1_vars)
    arg2_nvars = len(arg2_vars)
    arg1_degree = visitor.degree_map[arg1]
    arg2_degree = visitor.degree_map[arg2]

    if arg1_nvars > 1:
        arg1 = visitor.create_aux_var(arg1)
        arg1_vars = (arg1,)
        arg1_nvars = 1
        arg1_degree = 1

    if arg2_nvars > 1:
        arg2 = visitor.create_aux_var(arg2)
        arg2_vars = (arg2,)
        arg2_nvars = 1
        arg2_degree = 1

    res = arg1**arg2
    # at this point arg1 should have at most 1 variable 
    # and arg2 should have at most 1 variable
    if arg1_nvars == 0:
        visitor.node_to_var_map[res] = arg2_vars
    elif arg2_nvars == 0:
        visitor.node_to_var_map[res] = arg1_vars
    else:
        x = arg1_vars[0]
        y = arg2_vars[0]
        if x is y:
            visitor.node_to_var_map[res] = (x,)
        else:
            visitor.node_to_var_map[res] = (x, y)
    if arg1_degree == 0 and arg2_degree == 0:
        visitor.degree_map[res] = 0
    else:
        visitor.degree_map[res] = -1
    visitor.substitution_map[node] = res
    return res


def _handle_named_expression(node, data, visitor):
    assert len(data) == 1
    res = data[0]
    visitor.substitution_map[node] = res
    return res


def _handle_negation(node, data, visitor):
    arg = data[0]
    res = -arg
    visitor.node_to_var_map[res] = visitor.node_to_var_map[arg]
    visitor.degree_map[res] = visitor.degree_map[arg]
    visitor.substitution_map[node] = res
    return res


def _handle_unary(node, data, visitor):
    arg = data[0]
    arg_vars = visitor.node_to_var_map[arg]
    arg_nvars = len(arg_vars)
    arg_degree = visitor.degree_map[arg]

    if arg_nvars > 1:
        arg = visitor.create_aux_var(arg)
        arg_vars = (arg,)
        arg_nvars = 1
        arg_degree = 1
    res = node.create_node_with_local_data((arg,))
    visitor.node_to_var_map[res] = arg_vars
    if arg_degree == 0:
        visitor.degree_map[res] = 0
    else:
        visitor.degree_map[res] = -1
    visitor.substitution_map[node] = res
    return res


handlers = ExitNodeDispatcher()
handlers[VarData] = _handle_var
handlers[ScalarVar] = _handle_var
handlers[ParamData] = _handle_param
handlers[ScalarParam] = _handle_param
handlers[ProductExpression] = _handle_product
handlers[SumExpression] = _handle_sum
handlers[DivisionExpression] = _handle_division
handlers[PowExpression] = _handle_pow
handlers[MonomialTermExpression] = _handle_product
handlers[LinearExpression] = _handle_sum
handlers[ExpressionData] = _handle_named_expression
handlers[ScalarExpression] = _handle_named_expression
handlers[NegationExpression] = _handle_negation
handlers[UnaryFunctionExpression] = _handle_unary
handlers[int] = _handle_float
handlers[float] = _handle_float


class _UnivariateNonlinearDecompositionVisitor(StreamBasedExpressionVisitor):
    def __init__(self, **kwds):
        self.block = kwds.pop('aux_block')
        super().__init__(**kwds)
        self.node_to_var_map = ComponentMap()
        self.degree_map = ComponentMap()  # values will be 0 (constant), 1 (linear), or -1 (nonlinear)

        self.substitution_map = ComponentMap()

        self.block.x = VarList()
        self.block.c = ConstraintList()

    def initializeWalker(self, expr):
        if expr in self.substitution_map:
            return False, self.substitution_map[expr]
        return True, None
    
    def beforeChild(self, node, child, child_idx):
        if child in self.substitution_map:
            return False, self.substitution_map[child]
        return True, None
    
    def exitNode(self, node, data):
        nt = type(node)
        if nt in handlers:
            return handlers[type(node)](node, data, self)
        elif nt in native_numeric_types:
            handlers[nt] = _handle_float
            return _handle_float(node, data, self)
        else:
            raise NotImplementedError(f'unrecognized expression type: {nt}')

    def create_aux_var(self, expr):
        if expr in self.substitution_map:
            x = self.substitution_map[expr]
        else:
            x = self.block.x.add()
            self.substitution_map[expr] = x
            c = self.block.c.add(x == expr)
            # we need to compute bounds on x now because some of the 
            # handlers depend on variable bounds (e.g., division)
            fbbt(c)
        return x


@TransformationFactory.register(
    'contrib.piecewise.univariate_nonlinear_decomposition',
    doc="""
The purpose of this module/transformation is to convert any nonlinear model 
to the following form:

min/max e(x_i)*///**f(x_j) + b^T*x
s.t.
        g_j(x_i)*h_j(x_k) + a_j^T*x >/</== 0
        g_j(x_i)/h_j(x_k) + a_j^T*x >/</== 0
        g_j(x_i)**h_j(x_k) + a_j^T*x >/</== 0

By doing so, each nonlinear function is only a function of one or two variables. 
If this transformation is used prior to the nonlinear_to_pwl transformation, 
it can significantly reduce the complexity of the PWL approximation.
    """
)
class UnivariateNonlinearDecompositionTransformation(Transformation):
    def __init__(self):
        super().__init__()

    def _check_for_unknown_active_components(self, model):
        known_ctypes = {Constraint, Objective, Block}
        for ctype in model.collect_ctypes(active=True, descend_into=True):
            if not issubclass(ctype, ActiveComponent):
                continue
            if ctype in known_ctypes:
                continue
            if ctype is Suffix:
                continue
            raise NotImplementedError(
                f'UnivariateNonlinearDecompositionTransformation does not know how to '
                f'handle components with ctype {ctype}'
            )

    def _apply_to(self, model, **kwds):
        if kwds:
            raise ValueError('UnivariateNonlinearDecompositionTransformation does not take any keyword arguments')
        
        self._check_for_unknown_active_components(model)

        objectives = list(model.component_data_objects(Objective, active=True, descend_into=True))
        constraints = list(model.component_data_objects(Constraint, active=True, descend_into=True))
        
        bname = unique_component_name(model, 'auxiliary')
        setattr(model, bname, Block())
        block = getattr(model, bname)
        visitor = _UnivariateNonlinearDecompositionVisitor(aux_block=block)

        for con in constraints:
            lower, body, upper = con.to_bounded_expression(evaluate_bounds=True)
            new_body = visitor.walk_expression(body)
            if lower == upper:
                con.set_value(new_body == lower)
            else:
                con.set_value((lower, new_body, upper))

        for obj in objectives:
            new_expr = visitor.walk_expression(obj.expr)
            obj.expr = new_expr
