#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Utility functions and classes for the MindtPy solver."""
import logging
from pyomo.common.collections import ComponentMap
from pyomo.core import (
    Block,
    Constraint,
    VarList,
    Objective,
    Reals,
    Var,
    minimize,
    RangeSet,
    ConstraintList,
    TransformationFactory,
)
from pyomo.repn import generate_standard_repn
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available, McCormick
from pyomo.contrib.fbbt.fbbt import compute_bounds_on_expr
import pyomo.core.expr as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math

pyomo_nlp = attempt_import('pyomo.contrib.pynumero.interfaces.pyomo_nlp')[0]
numpy = attempt_import('numpy')[0]


def calc_jacobians(model, config):
    """Generates a map of jacobians for the variables in the model.

    This function generates a map of jacobians corresponding to the variables in the
    model.

    Parameters
    ----------
    model : Pyomo model
        Target model to calculate jacobian.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint w.r.t. variable)
    jacobians = ComponentMap()
    if config.differentiate_mode == 'reverse_symbolic':
        mode = EXPR.differentiate.Modes.reverse_symbolic
    elif config.differentiate_mode == 'sympy':
        mode = EXPR.differentiate.Modes.sympy
    for c in model.MindtPy_utils.nonlinear_constraint_list:
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = EXPR.differentiate(c.body, wrt_list=vars_in_constr, mode=mode)
        jacobians[c] = ComponentMap(
            (var, jac_wrt_var) for var, jac_wrt_var in zip(vars_in_constr, jac_list)
        )
    return jacobians


def initialize_feas_subproblem(m, config):
    """Adds feasibility slack variables according to config.feasibility_norm (given an infeasible problem).
       Defines the objective function of the feasibility subproblem.

    Parameters
    ----------
    m : Pyomo model
        The feasbility NLP subproblem.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    MindtPy = m.MindtPy_utils
    # generate new constraints
    for i, constr in enumerate(MindtPy.nonlinear_constraint_list, 1):
        if constr.has_ub():
            if config.feasibility_norm in {'L1', 'L2'}:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.upper <= MindtPy.feas_opt.slack_var[i]
                )
            else:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.upper <= MindtPy.feas_opt.slack_var
                )
        if constr.has_lb():
            if config.feasibility_norm in {'L1', 'L2'}:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.lower >= -MindtPy.feas_opt.slack_var[i]
                )
            else:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.lower >= -MindtPy.feas_opt.slack_var
                )
    # Setup objective function for the feasibility subproblem.
    if config.feasibility_norm == 'L1':
        MindtPy.feas_obj = Objective(
            expr=sum(s for s in MindtPy.feas_opt.slack_var.values()), sense=minimize
        )
    elif config.feasibility_norm == 'L2':
        MindtPy.feas_obj = Objective(
            expr=sum(s * s for s in MindtPy.feas_opt.slack_var.values()), sense=minimize
        )
    else:
        MindtPy.feas_obj = Objective(expr=MindtPy.feas_opt.slack_var, sense=minimize)
    MindtPy.feas_obj.deactivate()


def add_var_bound(model, config):
    """This function will add bounds for variables in nonlinear constraints if they are not bounded.

    This is to avoid an unbounded main problem in the LP/NLP algorithm. Thus, the model will be
    updated to include bounds for the unbounded variables in nonlinear constraints.

    Parameters
    ----------
    model : PyomoModel
        Target model to add bound for its variables.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    MindtPy = model.MindtPy_utils
    for c in MindtPy.nonlinear_constraint_list:
        for var in EXPR.identify_variables(c.body):
            if var.has_lb() and var.has_ub():
                continue
            elif not var.has_lb():
                if var.is_integer():
                    var.setlb(-config.integer_var_bound - 1)
                else:
                    var.setlb(-config.continuous_var_bound - 1)
            elif not var.has_ub():
                if var.is_integer():
                    var.setub(config.integer_var_bound)
                else:
                    var.setub(config.continuous_var_bound)


def generate_norm2sq_objective_function(model, setpoint_model, discrete_only=False):
    r"""This function generates objective (FP-NLP subproblem) for minimum
    euclidean distance to setpoint_model.

    L2 distance of (x,y) = \sqrt{\sum_i (x_i - y_i)^2}.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete
        variables, by default False.

    Returns
    -------
    Objective
        The norm2 square objective function.
    """
    # skip objective_value variable and slack_var variables
    var_filter = (
        (lambda v: v[1].is_integer())
        if discrete_only
        else (
            lambda v: 'MindtPy_utils.objective_value' not in v[1].name
            and 'MindtPy_utils.feas_opt.slack_var' not in v[1].name
        )
    )

    model_vars, setpoint_vars = zip(
        *filter(
            var_filter,
            zip(
                model.MindtPy_utils.variable_list,
                setpoint_model.MindtPy_utils.variable_list,
            ),
        )
    )
    assert len(model_vars) == len(
        setpoint_vars
    ), 'Trying to generate Squared Norm2 objective function for models with different number of variables'

    return Objective(
        expr=(
            sum(
                [
                    (model_var - setpoint_var.value) ** 2
                    for (model_var, setpoint_var) in zip(model_vars, setpoint_vars)
                ]
            )
        )
    )


def generate_norm1_objective_function(model, setpoint_model, discrete_only=False):
    r"""This function generates objective (PF-OA main problem) for minimum
    Norm1 distance to setpoint_model.

    Norm1 distance of (x,y) = \sum_i |x_i - y_i|.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete
        variables, by default False.

    Returns
    -------
    Objective
        The norm1 objective function.

    """
    # skip objective_value variable and slack_var variables
    var_filter = (
        (lambda v: v.is_integer())
        if discrete_only
        else (
            lambda v: 'MindtPy_utils.objective_value' not in v.name
            and 'MindtPy_utils.feas_opt.slack_var' not in v.name
        )
    )
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(
        setpoint_vars
    ), 'Trying to generate Norm1 objective function for models with different number of variables'
    model.MindtPy_utils.del_component('L1_obj')
    obj_block = model.MindtPy_utils.L1_obj = Block()
    obj_block.L1_obj_idx = RangeSet(len(model_vars))
    obj_block.L1_obj_var = Var(obj_block.L1_obj_idx, domain=Reals, bounds=(0, None))
    obj_block.abs_reform = ConstraintList()
    for idx, v_model, v_setpoint in zip(
        obj_block.L1_obj_idx, model_vars, setpoint_vars
    ):
        obj_block.abs_reform.add(
            expr=v_model - v_setpoint.value >= -obj_block.L1_obj_var[idx]
        )
        obj_block.abs_reform.add(
            expr=v_model - v_setpoint.value <= obj_block.L1_obj_var[idx]
        )

    return Objective(
        expr=sum(obj_block.L1_obj_var[idx] for idx in obj_block.L1_obj_idx)
    )


def generate_norm_inf_objective_function(model, setpoint_model, discrete_only=False):
    r"""This function generates objective (PF-OA main problem) for minimum Norm Infinity distance to setpoint_model.

    Norm-Infinity distance of (x,y) = \max_i |x_i - y_i|.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete variables, by default False.

    Returns
    -------
    Objective
        The norm infinity objective function.
    """
    # skip objective_value variable and slack_var variables
    var_filter = (
        (lambda v: v.is_integer())
        if discrete_only
        else (
            lambda v: 'MindtPy_utils.objective_value' not in v.name
            and 'MindtPy_utils.feas_opt.slack_var' not in v.name
        )
    )
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(
        setpoint_vars
    ), 'Trying to generate Norm Infinity objective function for models with different number of variables'
    model.MindtPy_utils.del_component('L_infinity_obj')
    obj_block = model.MindtPy_utils.L_infinity_obj = Block()
    obj_block.L_infinity_obj_var = Var(domain=Reals, bounds=(0, None))
    obj_block.abs_reform = ConstraintList()
    for v_model, v_setpoint in zip(model_vars, setpoint_vars):
        obj_block.abs_reform.add(
            expr=v_model - v_setpoint.value >= -obj_block.L_infinity_obj_var
        )
        obj_block.abs_reform.add(
            expr=v_model - v_setpoint.value <= obj_block.L_infinity_obj_var
        )

    return Objective(expr=obj_block.L_infinity_obj_var)


def generate_lag_objective_function(
    model, setpoint_model, config, timing, discrete_only=False
):
    """The function generates the second-order Taylor approximation of the Lagrangean.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    config : ConfigBlock
        The specific configurations for MindtPy.
    timing : Timing
        Timing
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete variables, by default False.

    Returns
    -------
    Objective
        The taylor extension(1st order or 2nd order) of the Lagrangean function.
    """
    temp_model = setpoint_model.clone()
    for var in temp_model.MindtPy_utils.variable_list:
        if var.is_integer():
            var.unfix()
    # objective_list[0] is the original objective function, not in MindtPy_utils block
    temp_model.MindtPy_utils.objective_list[0].activate()
    temp_model.MindtPy_utils.deactivate()
    TransformationFactory('core.relax_integer_vars').apply_to(temp_model)
    # Note: PyNumero does not support discrete variables
    # So PyomoNLP should operate on setpoint_model

    # Implementation 1
    # First calculate Jacobian and Hessian without assigning variable and constraint sequence, then use get_primal_indices to get the indices.
    with time_code(timing, 'PyomoNLP'):
        nlp = pyomo_nlp.PyomoNLP(temp_model)
        lam = [
            -temp_model.dual[constr]
            if abs(temp_model.dual[constr]) > config.zero_tolerance
            else 0
            for constr in nlp.get_pyomo_constraints()
        ]
        nlp.set_duals(lam)
        obj_grad = nlp.evaluate_grad_objective().reshape(-1, 1)
        jac = nlp.evaluate_jacobian().toarray()
        jac_lag = obj_grad + jac.transpose().dot(numpy.array(lam).reshape(-1, 1))
        jac_lag[abs(jac_lag) < config.zero_tolerance] = 0
        # jac_lag of continuous variables should be zero
        for var in temp_model.MindtPy_utils.continuous_variable_list:
            if 'MindtPy_utils.objective_value' not in var.name:
                jac_lag[nlp.get_primal_indices([var])[0]] = 0
        nlp_var = set([i.name for i in nlp.get_pyomo_variables()])
        first_order_term = sum(
            jac_lag[nlp.get_primal_indices([temp_var])[0]][0] * (var - temp_var.value)
            for var, temp_var in zip(
                model.MindtPy_utils.variable_list,
                temp_model.MindtPy_utils.variable_list,
            )
            if temp_var.name in nlp_var
        )

        if config.add_regularization == 'grad_lag':
            return Objective(expr=first_order_term, sense=minimize)
        elif config.add_regularization in {'hess_lag', 'hess_only_lag'}:
            # Implementation 1
            hess_lag = nlp.evaluate_hessian_lag().toarray()
            hess_lag[abs(hess_lag) < config.zero_tolerance] = 0
            second_order_term = 0.5 * sum(
                (var_i - temp_var_i.value)
                * float(
                    hess_lag[nlp.get_primal_indices([temp_var_i])[0]][
                        nlp.get_primal_indices([temp_var_j])[0]
                    ]
                )
                * (var_j - temp_var_j.value)
                for var_i, temp_var_i in zip(
                    model.MindtPy_utils.variable_list,
                    temp_model.MindtPy_utils.variable_list,
                )
                for var_j, temp_var_j in zip(
                    model.MindtPy_utils.variable_list,
                    temp_model.MindtPy_utils.variable_list,
                )
                if (temp_var_i.name in nlp_var and temp_var_j.name in nlp_var)
            )
            if config.add_regularization == 'hess_lag':
                return Objective(
                    expr=first_order_term + second_order_term, sense=minimize
                )
            elif config.add_regularization == 'hess_only_lag':
                return Objective(expr=second_order_term, sense=minimize)
        elif config.add_regularization == 'sqp_lag':
            var_filter = (
                (lambda v: v[1].is_integer())
                if discrete_only
                else (
                    lambda v: 'MindtPy_utils.objective_value' not in v[1].name
                    and 'MindtPy_utils.feas_opt.slack_var' not in v[1].name
                )
            )

            model_vars, setpoint_vars = zip(
                *filter(
                    var_filter,
                    zip(
                        model.MindtPy_utils.variable_list,
                        setpoint_model.MindtPy_utils.variable_list,
                    ),
                )
            )
            assert len(model_vars) == len(
                setpoint_vars
            ), 'Trying to generate Squared Norm2 objective function for models with different number of variables'
            if config.sqp_lag_scaling_coef is None:
                rho = 1
            elif config.sqp_lag_scaling_coef == 'fixed':
                r = 1
                rho = numpy.linalg.norm(jac_lag / (2 * r))
            elif config.sqp_lag_scaling_coef == 'variable_dependent':
                r = numpy.sqrt(len(temp_model.MindtPy_utils.discrete_variable_list))
                rho = numpy.linalg.norm(jac_lag / (2 * r))

            return Objective(
                expr=first_order_term
                + rho
                * sum(
                    [
                        (model_var - setpoint_var.value) ** 2
                        for (model_var, setpoint_var) in zip(model_vars, setpoint_vars)
                    ]
                )
            )


def generate_norm1_norm_constraint(model, setpoint_model, config, discrete_only=True):
    r"""This function generates constraint (PF-OA main problem) for minimum
    Norm1 distance to setpoint_model.

    Norm constraint is used to guarantees the monotonicity of the norm
    objective value sequence of all iterations.

    Norm1 distance of (x,y) = \sum_i |x_i - y_i|.
    Ref: Paper 'A storm of feasibility pumps for nonconvex MINLP' Eq. (16).

    Parameters
    ----------
    model : Pyomo model
        The model that needs the norm constraint.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    config : ConfigBlock
        The specific configurations for MindtPy.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete
        variables, by default True.

    """
    var_filter = (lambda v: v.is_integer()) if discrete_only else (lambda v: True)
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(
        setpoint_vars
    ), 'Trying to generate Norm1 norm constraint for models with different number of variables'
    norm_constraint_block = model.MindtPy_utils.L1_norm_constraint = Block()
    norm_constraint_block.L1_slack_idx = RangeSet(len(model_vars))
    norm_constraint_block.L1_slack_var = Var(
        norm_constraint_block.L1_slack_idx, domain=Reals, bounds=(0, None)
    )
    norm_constraint_block.abs_reform = ConstraintList()
    for idx, v_model, v_setpoint in zip(
        norm_constraint_block.L1_slack_idx, model_vars, setpoint_vars
    ):
        norm_constraint_block.abs_reform.add(
            expr=v_model - v_setpoint.value >= -norm_constraint_block.L1_slack_var[idx]
        )
        norm_constraint_block.abs_reform.add(
            expr=v_model - v_setpoint.value <= norm_constraint_block.L1_slack_var[idx]
        )
    rhs = config.fp_norm_constraint_coef * sum(
        abs(v_model.value - v_setpoint.value)
        for v_model, v_setpoint in zip(model_vars, setpoint_vars)
    )
    norm_constraint_block.sum_slack = Constraint(
        expr=sum(
            norm_constraint_block.L1_slack_var[idx]
            for idx in norm_constraint_block.L1_slack_idx
        )
        <= rhs
    )


def update_solver_timelimit(opt, solver_name, timing, config):
    """Updates the time limit for subsolvers.

    Parameters
    ----------
    opt : Solvers
        The solver object.
    solver_name : String
        The name of solver.
    timing : Timing
        Timing
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    elapsed = get_main_elapsed_time(timing)
    remaining = math.ceil(max(config.time_limit - elapsed, 1))
    if solver_name in {
        'cplex',
        'appsi_cplex',
        'cplex_persistent',
        'gurobi',
        'gurobi_persistent',
        'appsi_gurobi',
    }:
        opt.options['timelimit'] = remaining
    elif solver_name == 'appsi_highs':
        opt.config.time_limit = remaining
    elif solver_name == 'cyipopt':
        opt.config.options['max_cpu_time'] = float(remaining)
    elif solver_name == 'glpk':
        opt.options['tmlim'] = remaining
    elif solver_name == 'baron':
        opt.options['MaxTime'] = remaining
    elif solver_name in {'ipopt', 'appsi_ipopt'}:
        opt.options['max_cpu_time'] = remaining
    elif solver_name == 'gams':
        opt.options['add_options'].append('option Reslim=%s;' % remaining)


def set_solver_mipgap(opt, solver_name, config):
    """Set mipgap for subsolvers.

    Parameters
    ----------
    opt : Solvers
        The solver object.
    solver_name : String
        The name of solver.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if solver_name in {
        'cplex',
        'cplex_persistent',
        'gurobi',
        'gurobi_persistent',
        'appsi_gurobi',
        'glpk',
    }:
        opt.options['mipgap'] = config.mip_solver_mipgap
    elif solver_name == 'appsi_cplex':
        opt.options['mip_tolerances_mipgap'] = config.mip_solver_mipgap
    elif solver_name == 'appsi_highs':
        opt.config.mip_gap = config.mip_solver_mipgap
    elif solver_name == 'gams':
        opt.options['add_options'].append('option optcr=%s;' % config.mip_solver_mipgap)


def set_solver_constraint_violation_tolerance(opt, solver_name, config):
    """Set constraint violation tolerance for solvers.

    Parameters
    ----------
    opt : Solvers
        The solver object.
    solver_name : String
        The name of solver.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if solver_name == 'baron':
        opt.options['AbsConFeasTol'] = config.zero_tolerance
    elif solver_name in {'ipopt', 'appsi_ipopt'}:
        opt.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'cyipopt':
        opt.config.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'gams':
        if config.nlp_solver_args['solver'] in {
            'ipopt',
            'ipopth',
            'msnlp',
            'conopt',
            'baron',
        }:
            opt.options['add_options'].append('GAMS_MODEL.optfile=1')
            opt.options['add_options'].append(
                '$onecho > ' + config.nlp_solver_args['solver'] + '.opt'
            )
            if config.nlp_solver_args['solver'] in {'ipopt', 'ipopth'}:
                opt.options['add_options'].append(
                    'constr_viol_tol ' + str(config.zero_tolerance)
                )
                # Ipopt warmstart options
                opt.options['add_options'].append(
                    'warm_start_init_point       yes\n'
                    'warm_start_bound_push       1e-9\n'
                    'warm_start_bound_frac       1e-9\n'
                    'warm_start_slack_bound_frac 1e-9\n'
                    'warm_start_slack_bound_push 1e-9\n'
                    'warm_start_mult_bound_push  1e-9\n'
                )
            elif config.nlp_solver_args['solver'] == 'conopt':
                opt.options['add_options'].append(
                    'RTNWMA ' + str(config.zero_tolerance)
                )
            elif config.nlp_solver_args['solver'] == 'msnlp':
                opt.options['add_options'].append(
                    'feasibility_tolerance ' + str(config.zero_tolerance)
                )
            elif config.nlp_solver_args['solver'] == 'baron':
                opt.options['add_options'].append(
                    'AbsConFeasTol ' + str(config.zero_tolerance)
                )
            opt.options['add_options'].append('$offecho')


def get_integer_solution(model, string_zero=False):
    """Extract the value of integer variables from the provided model.

    Parameters
    ----------
    model : Pyomo model
        The model to extract value of integer variables.
    string_zero : bool, optional
        Whether to store zero as string, by default False.

    Returns
    -------
    tuple
        The tuple of integer variable values.
    """
    temp = []
    for var in model.MindtPy_utils.discrete_variable_list:
        if string_zero:
            if var.value == 0:
                # In CPLEX, negative zero is different from zero,
                # so we use string to denote this (Only in singletree).
                temp.append(str(var.value))
            else:
                temp.append(int(round(var.value)))
        else:
            temp.append(int(round(var.value)))
    return tuple(temp)


def copy_var_list_values_from_solution_pool(
    from_list,
    to_list,
    config,
    solver_model,
    var_map,
    solution_name,
    ignore_integrality=False,
):
    """Copy variable values from the solution pool to another list.

    Parameters
    ----------
    from_list : list
        The variables that provides the values to copy from.
    to_list : list
        The variables that need to set value.
    config : ConfigBlock
        The specific configurations for MindtPy.
    solver_model : solver model
        The solver model derived from pyomo model.
    var_map : dict
        The map of pyomo variables to solver variables.
    solution_name : int or str
        The name of the solution in the solution pool.
    ignore_integrality : bool, optional
        Whether to ignore the integrality of integer variables, by default False.
    """
    for v_from, v_to in zip(from_list, to_list):
        try:
            if config.mip_solver == 'cplex_persistent':
                var_val = solver_model.solution.pool.get_values(
                    solution_name, var_map[v_from]
                )
            elif config.mip_solver == 'gurobi_persistent':
                solver_model.setParam(gurobipy.GRB.Param.SolutionNumber, solution_name)
                var_val = var_map[v_from].Xn
            # We don't want to trigger the reset of the global stale
            # indicator, so we will set this variable to be "stale",
            # knowing that set_value will switch it back to "not
            # stale"
            v_to.stale = True
            # NOTE: PEP 2180 changes the var behavior so that domain /
            # bounds violations no longer generate exceptions (and
            # instead log warnings).  This means that the following will
            # always succeed and the ValueError should never be raised.
            v_to.set_value(var_val, skip_validation=True)
        except ValueError as e:
            config.logger.error(e)
            rounded_val = int(round(var_val))
            # Check to see if this is just a tolerance issue
            if ignore_integrality and v_to.is_integer():
                v_to.set_value(var_val, skip_validation=True)
            elif v_to.is_integer() and (
                abs(var_val - rounded_val) <= config.integer_tolerance
            ):
                v_to.set_value(rounded_val, skip_validation=True)
            elif abs(var_val) <= config.zero_tolerance and 0 in v_to.domain:
                v_to.set_value(0, skip_validation=True)
            else:
                config.logger.error(
                    'Unknown validation domain error setting variable %s' % (v_to.name,)
                )
                raise


class GurobiPersistent4MindtPy(GurobiPersistent):
    """A new persistent interface to Gurobi.

    Args:
        GurobiPersistent (PersistentSolver): A class that provides a persistent interface to Gurobi.
    """

    def _intermediate_callback(self):
        def f(gurobi_model, where):
            """Callback function for Gurobi.

            Args:
                gurobi_model (Gurobi model): the Gurobi model derived from pyomo model.
                where (int): an enum member of gurobipy.GRB.Callback.
            """
            self._callback_func(
                self._pyomo_model, self, where, self.mindtpy_solver, self.config
            )

        return f


def set_up_logger(config):
    """Set up the formatter and handler for logger.

    Parameters
    ----------
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    config.logger.handlers.clear()
    config.logger.propagate = False
    ch = logging.StreamHandler()
    ch.setLevel(config.logging_level)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    # add the handlers to logger
    config.logger.addHandler(ch)


def epigraph_reformulation(exp, slack_var_list, constraint_list, use_mcpp, sense):
    """Epigraph reformulation.

    Generate the epigraph reformulation for objective expressions.

    Parameters
    ----------
    slack_var_list : VarList
        Slack vars for epigraph reformulation.
    constraint_list : ConstraintList
        Epigraph constraint list.
    use_mcpp : Bool
        Whether to use mcpp to tighten the bound of slack variables.
    exp : Expression
        The expression to reformulate.
    sense : objective sense
        The objective sense.
    """
    slack_var = slack_var_list.add()
    if mcpp_available() and use_mcpp:
        mc_obj = McCormick(exp)
        slack_var.setub(mc_obj.upper())
        slack_var.setlb(mc_obj.lower())
    else:
        # Use Pyomo's contrib.fbbt package
        lb, ub = compute_bounds_on_expr(exp)
        if sense == minimize:
            slack_var.setlb(lb)
        else:
            slack_var.setub(ub)
    if sense == minimize:
        constraint_list.add(expr=slack_var >= exp)
    else:
        constraint_list.add(expr=slack_var <= exp)


def setup_results_object(results, model, config):
    """Record problem statistics for original model."""
    # Create the solver results object
    res = results
    prob = res.problem
    res.problem.name = model.name
    res.problem.number_of_nonzeros = None  # TODO
    res.solver.termination_condition = None
    res.solver.message = None
    res.solver.user_time = None
    res.solver.wallclock_time = None
    res.solver.termination_message = None
    # Record solver name
    res.solver.name = 'MindtPy' + str(config.strategy)

    num_of = build_model_size_report(model)

    # Get count of constraints and variables
    prob.number_of_constraints = num_of.activated.constraints
    prob.number_of_disjunctions = num_of.activated.disjunctions
    prob.number_of_variables = num_of.activated.variables
    prob.number_of_binary_variables = num_of.activated.binary_variables
    prob.number_of_continuous_variables = num_of.activated.continuous_variables
    prob.number_of_integer_variables = num_of.activated.integer_variables

    config.logger.info(
        "Original model has %s constraints (%s nonlinear) "
        "and %s disjunctions, "
        "with %s variables, of which %s are binary, %s are integer, "
        "and %s are continuous."
        % (
            num_of.activated.constraints,
            num_of.activated.nonlinear_constraints,
            num_of.activated.disjunctions,
            num_of.activated.variables,
            num_of.activated.binary_variables,
            num_of.activated.integer_variables,
            num_of.activated.continuous_variables,
        )
    )
    config.logger.info(
        '{} is the initial strategy being used.\n'.format(config.init_strategy)
    )
    config.logger.info(
        ' ==============================================================================================='
    )
    config.logger.info(
        ' {:>9} | {:>15} | {:>15} | {:>12} | {:>12} | {:^7} | {:>7}\n'.format(
            'Iteration',
            'Subproblem Type',
            'Objective Value',
            'Primal Bound',
            'Dual Bound',
            ' Gap ',
            'Time(s)',
        )
    )


def fp_converged(working_model, mip_model, proj_zero_tolerance, discrete_only=True):
    """Calculates the euclidean norm between the discrete variables in the MIP and NLP models.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    proj_zero_tolerance : Float
        The projection zero tolerance of Feasibility Pump.
    discrete_only : bool, optional
        Whether to only optimize on distance between the discrete variables, by default True.

    Returns
    -------
    distance : float
        The euclidean norm between the discrete variables in the MIP and NLP models.
    """
    distance = max(
        (nlp_var.value - milp_var.value) ** 2
        for (nlp_var, milp_var) in zip(
            working_model.MindtPy_utils.variable_list,
            mip_model.MindtPy_utils.variable_list,
        )
        if (not discrete_only) or milp_var.is_integer()
    )
    return distance <= proj_zero_tolerance


def add_orthogonality_cuts(working_model, mip_model, config):
    """Add orthogonality cuts.

    This function adds orthogonality cuts to avoid cycling when the independence constraint qualification is not satisfied.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    mip_integer_vars = mip_model.MindtPy_utils.discrete_variable_list
    nlp_integer_vars = working_model.MindtPy_utils.discrete_variable_list
    orthogonality_cut = (
        sum(
            (nlp_v.value - mip_v.value) * (mip_v - nlp_v.value)
            for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)
        )
        >= 0
    )
    mip_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(orthogonality_cut)
    if config.fp_projcuts:
        orthogonality_cut = (
            sum(
                (nlp_v.value - mip_v.value) * (nlp_v - nlp_v.value)
                for mip_v, nlp_v in zip(mip_integer_vars, nlp_integer_vars)
            )
            >= 0
        )
        working_model.MindtPy_utils.cuts.fp_orthogonality_cuts.add(orthogonality_cut)


def generate_norm_constraint(fp_nlp_model, mip_model, config):
    """Generate the norm constraint for the FP-NLP subproblem.

    Parameters
    ----------
    fp_nlp_model : Pyomo model
        The feasibility pump NLP subproblem.
    mip_model : Pyomo model
        The mip_model model.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    if config.fp_main_norm == 'L1':
        # TODO: check if we can access the block defined in FP-main problem
        generate_norm1_norm_constraint(
            fp_nlp_model, mip_model, config, discrete_only=True
        )
    elif config.fp_main_norm == 'L2':
        fp_nlp_model.norm_constraint = Constraint(
            expr=sum(
                (nlp_var - mip_var.value) ** 2
                - config.fp_norm_constraint_coef * (nlp_var.value - mip_var.value) ** 2
                for nlp_var, mip_var in zip(
                    fp_nlp_model.MindtPy_utils.discrete_variable_list,
                    mip_model.MindtPy_utils.discrete_variable_list,
                )
            )
            <= 0
        )
    elif config.fp_main_norm == 'L_infinity':
        fp_nlp_model.norm_constraint = ConstraintList()
        rhs = config.fp_norm_constraint_coef * max(
            nlp_var.value - mip_var.value
            for nlp_var, mip_var in zip(
                fp_nlp_model.MindtPy_utils.discrete_variable_list,
                mip_model.MindtPy_utils.discrete_variable_list,
            )
        )
        for nlp_var, mip_var in zip(
            fp_nlp_model.MindtPy_utils.discrete_variable_list,
            mip_model.MindtPy_utils.discrete_variable_list,
        ):
            fp_nlp_model.norm_constraint.add(nlp_var - mip_var.value <= rhs)
