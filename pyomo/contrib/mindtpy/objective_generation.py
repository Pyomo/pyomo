"""Generates objective functions like L1, L2 and Linf distance"""

from pyomo.core import (Var, Objective)


def generate_L2_objective_function(model, setpoint_model, discrete_only=False):
    """Generate objective for minimum euclidean distance to setpoint_model

    L2 distance of (x,y) = \sqrt{\sum_i (x_i - y_i)^2}

    discrete_only -- only optimize on distance between the discrete variables
    """
    var_filter = (lambda v: not v.is_continuous()) if discrete_only \
                 else (lambda v: True)
    model_vars = filter(var_filter, model.component_data_objects(Var))
    setpoint_vars = filter(var_filter, setpoint_model.component_data_objects(Var))
    assert len(model_vars) == len(setpoint_vars), "Trying to generate L1 objective function for models with different number of variables"

    return Objective(expr=(
        sum([(model_var - setpoint_var.value)**2
             for (model_var, setpoint_var) in
             zip(model_vars, setpoint_vars)])))


def generate_L1_objective_function(model, setpoint_model, discrete_only=False):
    """Generate objective for minimum L1 distance to setpoint model

    L1 distance of (x,y) = \sum_i |x_i - y_i|

    discrete_only -- only optimize on distance between the discrete variables
    """

    var_filter = (lambda v: not v.is_continuous()) if discrete_only \
                 else (lambda v: True)
    model_vars = filter(var_filter, model.component_data_objects(Var))
    setpoint_vars = filter(var_filter, setpoint_model.component_data_objects(Var))
    assert len(model_vars) == len(setpoint_vars), "Trying to generate L1 objective function for models with different number of variables"

    model.L1_obj_var = Var(domain=Reals, bounds=(0, None))
    model.L1_obj_fun = Objective(expr=model.L1_obj_var)
    model.L1_obj_ub_idx = RangeSet(
        len(model_vars))
    model.L1_obj_ub_constr = Constraint(
        model.L1_obj_ub_idx, rule=lambda i: model.L1_obj_var >= 0)
    model.L1_obj_lb_idx = RangeSet(
        len(model_vars))
    model.L1_obj_lb_constr = Constraint(
        model.L1_obj_lb_idx, rule=lambda i: model.L1_obj_var >= 0)  # 'empty' constraint (will be set later)

    for (c_lb, c_ub, v_model, v_setpoint) in zip(model.L1_obj_lb_idx,
                                                 model.L1_obj_ub_idx,
                                                 model_vars,
                                                 setpoint_vars):
        model.L1_obj_lb_constr[c_lb].set_value(
            expr=v_model - v_setpoint.value >= -model.L1_obj_var)
        model.L1_obj_ub_constr[c_ub].set_value(
            expr=v_model - v_setpoint.value <= model.L1_obj_var)
