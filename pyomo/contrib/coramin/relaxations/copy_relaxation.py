from .mccormick import PWMcCormickRelaxationData, PWMcCormickRelaxation
from .univariate import PWXSquaredRelaxationData, PWUnivariateRelaxationData, PWArctanRelaxationData, \
    PWCosRelaxationData, PWSinRelaxationData
from .univariate import PWXSquaredRelaxation, PWUnivariateRelaxation, PWArctanRelaxation, \
    PWCosRelaxation, PWSinRelaxation
from .alphabb import AlphaBBRelaxation, AlphaBBRelaxationData
from .multivariate import MultivariateRelaxationData, MultivariateRelaxation
from pyomo.core.expr.visitor import replace_expressions
from coramin.utils.coramin_enums import FunctionShape


def copy_relaxation_with_local_data(rel, old_var_to_new_var_map):
    """
    This function copies a relaxation object with new variables.
    Note that only what can be set through the set_input and build
    methods are copied. For example, piecewise partitioning points
    are not copied.

    Parameters
    ----------
    rel: coramin.relaxations.relaxations_base.BaseRelaxationData
        The relaxation to be copied
    old_var_to_new_var_map: dict
        Map from the original variable id to the new variable

    Returns
    -------
    rel: coramin.relaxations.relaxations_base.BaseRelaxationData
        The copy of rel with new variables
    """
    if isinstance(rel, PWXSquaredRelaxationData):
        new_x = old_var_to_new_var_map[id(rel.get_rhs_vars()[0])]
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_rel = PWXSquaredRelaxation(concrete=True)
        new_rel.set_input(x=new_x, aux_var=new_aux_var, pw_repn=rel._pw_repn,
                          use_linear_relaxation=rel.use_linear_relaxation,
                          relaxation_side=rel.relaxation_side)
    elif isinstance(rel, PWArctanRelaxationData):
        new_x = old_var_to_new_var_map[id(rel.get_rhs_vars()[0])]
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_rel = PWArctanRelaxation(concrete=True)
        new_rel.set_input(x=new_x, aux_var=new_aux_var, pw_repn=rel._pw_repn,
                          relaxation_side=rel.relaxation_side,
                          use_linear_relaxation=rel.use_linear_relaxation)
    elif isinstance(rel, PWSinRelaxationData):
        new_x = old_var_to_new_var_map[id(rel.get_rhs_vars()[0])]
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_rel = PWSinRelaxation(concrete=True)
        new_rel.set_input(x=new_x, aux_var=new_aux_var, pw_repn=rel._pw_repn,
                          relaxation_side=rel.relaxation_side,
                          use_linear_relaxation=rel.use_linear_relaxation)
    elif isinstance(rel, PWCosRelaxationData):
        new_x = old_var_to_new_var_map[id(rel.get_rhs_vars()[0])]
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_rel = PWCosRelaxation(concrete=True)
        new_rel.set_input(x=new_x, aux_var=new_aux_var, pw_repn=rel._pw_repn,
                          relaxation_side=rel.relaxation_side,
                          use_linear_relaxation=rel.use_linear_relaxation)
    elif isinstance(rel, PWUnivariateRelaxationData):
        new_x = old_var_to_new_var_map[id(rel.get_rhs_vars()[0])]
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_f_x_expr = replace_expressions(rel.get_rhs_expr(),
                                           substitution_map=old_var_to_new_var_map,
                                           remove_named_expressions=True)
        new_rel = PWUnivariateRelaxation(concrete=True)
        if rel.is_rhs_convex():
            shape = FunctionShape.CONVEX
        elif rel.is_rhs_concave():
            shape = FunctionShape.CONCAVE
        else:
            shape = FunctionShape.UNKNOWN
        new_rel.set_input(x=new_x, aux_var=new_aux_var, shape=shape,
                          f_x_expr=new_f_x_expr, pw_repn=rel._pw_repn,
                          relaxation_side=rel.relaxation_side,
                          use_linear_relaxation=rel.use_linear_relaxation)
    elif isinstance(rel, PWMcCormickRelaxationData):
        rhs_vars = rel.get_rhs_vars()
        old_x1 = rhs_vars[0]
        old_x2 = rhs_vars[1]
        new_x1 = old_var_to_new_var_map[id(old_x1)]
        new_x2 = old_var_to_new_var_map[id(old_x2)]
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_rel = PWMcCormickRelaxation(concrete=True)
        new_rel.set_input(x1=new_x1, x2=new_x2, aux_var=new_aux_var,
                          relaxation_side=rel.relaxation_side)
    elif isinstance(rel, AlphaBBRelaxationData):
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        new_f_x_expr = replace_expressions(
            rel.get_rhs_expr(),
            substitution_map=old_var_to_new_var_map,
            remove_named_expressions=True
        )
        new_rel = AlphaBBRelaxation(concrete=True)
        new_rel.set_input(
            aux_var=new_aux_var,
            f_x_expr=new_f_x_expr,
            relaxation_side=rel.relaxation_side,
            use_linear_relaxation=rel.use_linear_relaxation,
            eigenvalue_bounder=rel.hessian.method,
            eigenvalue_opt=rel.hessian.opt,
        )
    elif isinstance(rel, MultivariateRelaxationData):
        new_aux_var = old_var_to_new_var_map[id(rel.get_aux_var())]
        if rel.is_rhs_convex():
            shape = FunctionShape.CONVEX
        elif rel.is_rhs_concave():
            shape = FunctionShape.CONCAVE
        else:
            shape = FunctionShape.UNKNOWN
        new_f_x_expr = replace_expressions(rel.get_rhs_expr(),
                                           substitution_map=old_var_to_new_var_map,
                                           remove_named_expressions=True)
        new_rel = MultivariateRelaxation(concrete=True)
        new_rel.set_input(aux_var=new_aux_var, shape=shape, f_x_expr=new_f_x_expr,
                          use_linear_relaxation=rel.use_linear_relaxation)
    else:
        raise ValueError('Unrecognized relaxation: {0}'.format(str(type(rel))))

    new_rel.small_coef = rel.small_coef
    new_rel.large_coef = rel.large_coef
    new_rel.safety_tol = rel.safety_tol

    return new_rel
