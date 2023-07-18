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
from pyomo.core.expr import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.opt import ProblemSense
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.util.model_size import build_model_size_report
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.common.dependencies import attempt_import
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math

pyomo_nlp = attempt_import('pyomo.contrib.pynumero.interfaces.pyomo_nlp')[0]
numpy = attempt_import('numpy')[0]


class MindtPySolveData(object):
    """Data container to hold solve-instance data."""

    pass


def calc_jacobians(model, config):
    """Generates a map of jacobians for the variables in the model.

    This function generates a map of jacobians corresponding to the variables in the
    model and adds this ComponentMap to solve_data.

    Parameters
    ----------
    model : Pyomo model
        Target model to calculate jacobian.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint wrt. variable)
    jacobians = ComponentMap()
    if config.differentiate_mode == 'reverse_symbolic':
        mode = differentiate.Modes.reverse_symbolic
    elif config.differentiate_mode == 'sympy':
        mode = differentiate.Modes.sympy
    for c in model.MindtPy_utils.nonlinear_constraint_list:
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = differentiate(c.body, wrt_list=vars_in_constr, mode=mode)
        jacobians[c] = ComponentMap(
            (var, jac_wrt_var) for var, jac_wrt_var in zip(vars_in_constr, jac_list)
        )
    return jacobians


def add_feas_slacks(m, config):
    """Adds feasibility slack variables according to config.feasibility_norm (given an infeasible problem).

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
            float(jac_lag[nlp.get_primal_indices([temp_var])[0]])
            * (var - temp_var.value)
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


def set_solver_options(opt, timing, config, solver_type, regularization=False):
    """Set options for MIP/NLP solvers.

    Parameters
    ----------
    opt : SolverFactory
        The MIP/NLP solver.
    timing : Timing
        Timing.
    config : ConfigBlock
        The specific configurations for MindtPy.
    solver_type : str
        The type of the solver, i.e. mip or nlp.
    regularization : bool, optional
        Whether the solver is used to solve the regularization problem, by default False.
    """
    # TODO: integrate nlp_args here
    # nlp_args = dict(config.nlp_solver_args)
    elapsed = get_main_elapsed_time(timing)
    remaining = int(max(config.time_limit - elapsed, 1))
    if solver_type == 'mip':
        if regularization:
            solver_name = config.mip_regularization_solver
            if config.regularization_mip_threads > 0:
                opt.options['threads'] = config.regularization_mip_threads
        else:
            solver_name = config.mip_solver
            if config.threads > 0:
                opt.options['threads'] = config.threads
    elif solver_type == 'nlp':
        solver_name = config.nlp_solver
    # TODO: opt.name doesn't work for GAMS
    if solver_name in {'cplex', 'gurobi', 'gurobi_persistent', 'appsi_gurobi'}:
        opt.options['timelimit'] = remaining
        opt.options['mipgap'] = config.mip_solver_mipgap
        if solver_name == 'gurobi_persistent' and config.single_tree:
            # PreCrush: Controls presolve reductions that affect user cuts
            # You should consider setting this parameter to 1 if you are using callbacks to add your own cuts.
            opt.options['PreCrush'] = 1
            opt.options['LazyConstraints'] = 1
        if regularization == True:
            if solver_name == 'cplex':
                if config.solution_limit is not None:
                    opt.options['mip limits solutions'] = config.solution_limit
                opt.options['mip strategy presolvenode'] = 3
                # TODO: need to discuss if this option should be added.
                if config.add_regularization in {'hess_lag', 'hess_only_lag'}:
                    opt.options['optimalitytarget'] = 3
            elif solver_name == 'gurobi':
                if config.solution_limit is not None:
                    opt.options['SolutionLimit'] = config.solution_limit
                opt.options['Presolve'] = 2
    elif solver_name == 'cplex_persistent':
        opt.options['timelimit'] = remaining
        opt._solver_model.parameters.mip.tolerances.mipgap.set(config.mip_solver_mipgap)
        if regularization is True:
            if config.solution_limit is not None:
                opt._solver_model.parameters.mip.limits.solutions.set(
                    config.solution_limit
                )
            opt._solver_model.parameters.mip.strategy.presolvenode.set(3)
            if config.add_regularization in {'hess_lag', 'hess_only_lag'}:
                opt._solver_model.parameters.optimalitytarget.set(3)
    elif solver_name == 'appsi_cplex':
        opt.options['timelimit'] = remaining
        opt.options['mip_tolerances_mipgap'] = config.mip_solver_mipgap
        if regularization is True:
            if config.solution_limit is not None:
                opt.options['mip_limits_solutions'] = config.solution_limit
            opt.options['mip_strategy_presolvenode'] = 3
            if config.add_regularization in {'hess_lag', 'hess_only_lag'}:
                opt.options['optimalitytarget'] = 3
    elif solver_name == 'glpk':
        opt.options['tmlim'] = remaining
        opt.options['mipgap'] = config.mip_solver_mipgap
    elif solver_name == 'baron':
        opt.options['MaxTime'] = remaining
        opt.options['AbsConFeasTol'] = config.zero_tolerance
    elif solver_name in {'ipopt', 'appsi_ipopt'}:
        opt.options['max_cpu_time'] = remaining
        opt.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'cyipopt':
        opt.config.options['max_cpu_time'] = float(remaining)
        opt.config.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'gams':
        if solver_type == 'mip':
            opt.options['add_options'] = [
                'option optcr=%s;' % config.mip_solver_mipgap,
                'option reslim=%s;' % remaining,
            ]
        elif solver_type == 'nlp':
            opt.options['add_options'] = ['option reslim=%s;' % remaining]
            if config.nlp_solver_args.__contains__('solver'):
                if config.nlp_solver_args['solver'] in {
                    'ipopt',
                    'ipopth',
                    'msnlp',
                    'conopt',
                    'baron',
                }:
                    if config.nlp_solver_args['solver'] == 'ipopt':
                        opt.options['add_options'].append('$onecho > ipopt.opt')
                        opt.options['add_options'].append(
                            'constr_viol_tol ' + str(config.zero_tolerance)
                        )
                    elif config.nlp_solver_args['solver'] == 'ipopth':
                        opt.options['add_options'].append('$onecho > ipopth.opt')
                        opt.options['add_options'].append(
                            'constr_viol_tol ' + str(config.zero_tolerance)
                        )
                        # TODO: Ipopt warmstart option
                        # opt.options['add_options'].append('warm_start_init_point       yes\n'
                        #                                   'warm_start_bound_push       1e-9\n'
                        #                                   'warm_start_bound_frac       1e-9\n'
                        #                                   'warm_start_slack_bound_frac 1e-9\n'
                        #                                   'warm_start_slack_bound_push 1e-9\n'
                        #                                   'warm_start_mult_bound_push  1e-9\n')
                    elif config.nlp_solver_args['solver'] == 'conopt':
                        opt.options['add_options'].append('$onecho > conopt.opt')
                        opt.options['add_options'].append(
                            'RTNWMA ' + str(config.zero_tolerance)
                        )
                    elif config.nlp_solver_args['solver'] == 'msnlp':
                        opt.options['add_options'].append('$onecho > msnlp.opt')
                        opt.options['add_options'].append(
                            'feasibility_tolerance ' + str(config.zero_tolerance)
                        )
                    elif config.nlp_solver_args['solver'] == 'baron':
                        opt.options['add_options'].append('$onecho > baron.opt')
                        opt.options['add_options'].append(
                            'AbsConFeasTol ' + str(config.zero_tolerance)
                        )
                    opt.options['add_options'].append('$offecho')
                    opt.options['add_options'].append('GAMS_MODEL.optfile=1')


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
                # In cplex, negative zero is different from zero, so we use string to denote this(Only in singletree)
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
        except ValueError as err:
            err_msg = getattr(err, 'message', str(err))
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
                gurobi_model (gurobi model): the gurobi model derived from pyomo model.
                where (int): an enum member of gurobipy.GRB.Callback.
            """
            self._callback_func(
                self._pyomo_model, self, where, self.solve_data, self.config
            )

        return f


def update_gap(solve_data):
    """Update the relative gap and the absolute gap.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    """
    if solve_data.objective_sense == minimize:
        solve_data.abs_gap = solve_data.primal_bound - solve_data.dual_bound
    else:
        solve_data.abs_gap = solve_data.dual_bound - solve_data.primal_bound
    solve_data.rel_gap = solve_data.abs_gap / (abs(solve_data.primal_bound) + 1e-10)


def update_dual_bound(solve_data, bound_value):
    """Update the dual bound.

    Call after solving relaxed problem, including relaxed NLP and MIP main problem.
    Use the optimal primal bound of the relaxed problem to update the dual bound.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    bound_value : float
        The input value used to update the dual bound.
    """
    if math.isnan(bound_value):
        return
    if solve_data.objective_sense == minimize:
        solve_data.dual_bound = max(bound_value, solve_data.dual_bound)
        solve_data.dual_bound_improved = (
            solve_data.dual_bound > solve_data.dual_bound_progress[-1]
        )
    else:
        solve_data.dual_bound = min(bound_value, solve_data.dual_bound)
        solve_data.dual_bound_improved = (
            solve_data.dual_bound < solve_data.dual_bound_progress[-1]
        )
    solve_data.dual_bound_progress.append(solve_data.dual_bound)
    solve_data.dual_bound_progress_time.append(get_main_elapsed_time(solve_data.timing))
    if solve_data.dual_bound_improved:
        update_gap(solve_data)


def update_suboptimal_dual_bound(solve_data, results):
    """If the relaxed problem is not solved to optimality, the dual bound is updated
    according to the dual bound of relaxed problem.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    results : SolverResults
        Results from solving the relaxed problem.
        The dual bound of the relaxed problem can only be obtained from the result object.
    """
    if solve_data.objective_sense == minimize:
        bound_value = results.problem.lower_bound
    else:
        bound_value = results.problem.upper_bound
    update_dual_bound(solve_data, bound_value)


def update_primal_bound(solve_data, bound_value):
    """Update the primal bound.

    Call after solve fixed NLP subproblem.
    Use the optimal primal bound of the relaxed problem to update the dual bound.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    bound_value : float
        The input value used to update the primal bound.
    """
    if math.isnan(bound_value):
        return
    if solve_data.objective_sense == minimize:
        solve_data.primal_bound = min(bound_value, solve_data.primal_bound)
        solve_data.primal_bound_improved = (
            solve_data.primal_bound < solve_data.primal_bound_progress[-1]
        )
    else:
        solve_data.primal_bound = max(bound_value, solve_data.primal_bound)
        solve_data.primal_bound_improved = (
            solve_data.primal_bound > solve_data.primal_bound_progress[-1]
        )
    solve_data.primal_bound_progress.append(solve_data.primal_bound)
    solve_data.primal_bound_progress_time.append(
        get_main_elapsed_time(solve_data.timing)
    )
    if solve_data.primal_bound_improved:
        update_gap(solve_data)


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


def get_dual_integral(solve_data, config):
    """Calculate the dual integral.
    Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.

    Returns
    -------
    float
        The dual integral.
    """
    dual_integral = 0
    dual_bound_progress = solve_data.dual_bound_progress.copy()
    # Initial dual bound is set to inf or -inf. To calculate dual integral, we set
    # initial_dual_bound to 10% greater or smaller than the first_found_dual_bound.
    # TODO: check if the calculation of initial_dual_bound needs to be modified.
    for dual_bound in dual_bound_progress:
        if dual_bound != dual_bound_progress[0]:
            break
    for i in range(len(dual_bound_progress)):
        if dual_bound_progress[i] == solve_data.dual_bound_progress[0]:
            dual_bound_progress[i] = dual_bound * (
                1
                - config.initial_bound_coef
                * solve_data.objective_sense
                * math.copysign(1, dual_bound)
            )
        else:
            break
    for i in range(len(dual_bound_progress)):
        if i == 0:
            dual_integral += abs(dual_bound_progress[i] - solve_data.dual_bound) * (
                solve_data.dual_bound_progress_time[i]
            )
        else:
            dual_integral += abs(dual_bound_progress[i] - solve_data.dual_bound) * (
                solve_data.dual_bound_progress_time[i]
                - solve_data.dual_bound_progress_time[i - 1]
            )
    config.logger.info(' {:<25}:   {:>7.4f} '.format('Dual integral', dual_integral))
    return dual_integral


def get_primal_integral(solve_data, config):
    """Calculate the primal integral.
    Ref: The confined primal integral. [http://www.optimization-online.org/DB_FILE/2020/07/7910.pdf]

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.

    Returns
    -------
    float
        The primal integral.
    """
    primal_integral = 0
    primal_bound_progress = solve_data.primal_bound_progress.copy()
    # Initial primal bound is set to inf or -inf. To calculate primal integral, we set
    # initial_primal_bound to 10% greater or smaller than the first_found_primal_bound.
    # TODO: check if the calculation of initial_primal_bound needs to be modified.
    for primal_bound in primal_bound_progress:
        if primal_bound != primal_bound_progress[0]:
            break
    for i in range(len(primal_bound_progress)):
        if primal_bound_progress[i] == solve_data.primal_bound_progress[0]:
            primal_bound_progress[i] = primal_bound * (
                1
                + config.initial_bound_coef
                * solve_data.objective_sense
                * math.copysign(1, primal_bound)
            )
        else:
            break
    for i in range(len(primal_bound_progress)):
        if i == 0:
            primal_integral += abs(
                primal_bound_progress[i] - solve_data.primal_bound
            ) * (solve_data.primal_bound_progress_time[i])
        else:
            primal_integral += abs(
                primal_bound_progress[i] - solve_data.primal_bound
            ) * (
                solve_data.primal_bound_progress_time[i]
                - solve_data.primal_bound_progress_time[i - 1]
            )

    config.logger.info(
        ' {:<25}:   {:>7.4f} '.format('Primal integral', primal_integral)
    )
    return primal_integral


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


def process_objective(
    solve_data,
    config,
    move_objective=False,
    use_mcpp=False,
    update_var_con_list=True,
    partition_nonlinear_terms=True,
    obj_handleable_polynomial_degree={0, 1},
    constr_handleable_polynomial_degree={0, 1},
):
    """Process model objective function.
    Check that the model has only 1 valid objective.
    If the objective is nonlinear, move it into the constraints.
    If no objective function exists, emit a warning and create a dummy
    objective.
    Parameters
    ----------
    solve_data (MindtPySolveData): solver environment data class
    config (ConfigBlock): solver configuration options
    move_objective (bool): if True, move even linear
        objective functions to the constraints
    update_var_con_list (bool): if True, the variable/constraint/objective lists will not be updated.
        This arg is set to True by default. Currently, update_var_con_list will be set to False only when
        add_regularization is not None in MindtPy.
    partition_nonlinear_terms (bool): if True, partition sum of nonlinear terms in the objective function.
    """
    m = solve_data.working_model
    util_block = getattr(m, solve_data.util_block_name)
    # Handle missing or multiple objectives
    active_objectives = list(
        m.component_data_objects(ctype=Objective, active=True, descend_into=True)
    )
    solve_data.results.problem.number_of_objectives = len(active_objectives)
    if len(active_objectives) == 0:
        config.logger.warning('Model has no active objectives. Adding dummy objective.')
        util_block.dummy_objective = Objective(expr=1)
        main_obj = util_block.dummy_objective
    elif len(active_objectives) > 1:
        raise ValueError('Model has multiple active objectives.')
    else:
        main_obj = active_objectives[0]
    solve_data.results.problem.sense = (
        ProblemSense.minimize if main_obj.sense == 1 else ProblemSense.maximize
    )
    solve_data.objective_sense = main_obj.sense

    # Move the objective to the constraints if it is nonlinear or move_objective is True.
    if (
        main_obj.expr.polynomial_degree() not in obj_handleable_polynomial_degree
        or move_objective
    ):
        if move_objective:
            config.logger.info("Moving objective to constraint set.")
        else:
            config.logger.info("Objective is nonlinear. Moving it to constraint set.")
        util_block.objective_value = VarList(domain=Reals, initialize=0)
        util_block.objective_constr = ConstraintList()
        if (
            main_obj.expr.polynomial_degree() not in obj_handleable_polynomial_degree
            and partition_nonlinear_terms
            and main_obj.expr.__class__ is EXPR.SumExpression
        ):
            repn = generate_standard_repn(
                main_obj.expr, quadratic=2 in obj_handleable_polynomial_degree
            )
            # the following code will also work if linear_subexpr is a constant.
            linear_subexpr = (
                repn.constant
                + sum(
                    coef * var for coef, var in zip(repn.linear_coefs, repn.linear_vars)
                )
                + sum(
                    coef * var1 * var2
                    for coef, (var1, var2) in zip(
                        repn.quadratic_coefs, repn.quadratic_vars
                    )
                )
            )
            # only need to generate one epigraph constraint for the sum of all linear terms and constant
            epigraph_reformulation(
                linear_subexpr,
                util_block.objective_value,
                util_block.objective_constr,
                use_mcpp,
                main_obj.sense,
            )
            nonlinear_subexpr = repn.nonlinear_expr
            if nonlinear_subexpr.__class__ is EXPR.SumExpression:
                for subsubexpr in nonlinear_subexpr.args:
                    epigraph_reformulation(
                        subsubexpr,
                        util_block.objective_value,
                        util_block.objective_constr,
                        use_mcpp,
                        main_obj.sense,
                    )
            else:
                epigraph_reformulation(
                    nonlinear_subexpr,
                    util_block.objective_value,
                    util_block.objective_constr,
                    use_mcpp,
                    main_obj.sense,
                )
        else:
            epigraph_reformulation(
                main_obj.expr,
                util_block.objective_value,
                util_block.objective_constr,
                use_mcpp,
                main_obj.sense,
            )

        main_obj.deactivate()
        util_block.objective = Objective(
            expr=sum(util_block.objective_value[:]), sense=main_obj.sense
        )

        if (
            main_obj.expr.polynomial_degree() not in obj_handleable_polynomial_degree
            or (move_objective and update_var_con_list)
        ):
            util_block.variable_list.extend(util_block.objective_value[:])
            util_block.continuous_variable_list.extend(util_block.objective_value[:])
            util_block.constraint_list.extend(util_block.objective_constr[:])
            util_block.objective_list.append(util_block.objective)
            for constr in util_block.objective_constr[:]:
                if (
                    constr.body.polynomial_degree()
                    in constr_handleable_polynomial_degree
                ):
                    util_block.linear_constraint_list.append(constr)
                else:
                    util_block.nonlinear_constraint_list.append(constr)


def fp_converged(working_model, mip_model, config, discrete_only=True):
    """Calculates the euclidean norm between the discrete variables in the MIP and NLP models.

    Parameters
    ----------
    working_model : Pyomo model
        The working model(original model).
    mip_model : Pyomo model
        The mip model.
    config : ConfigBlock
        The specific configurations for MindtPy.
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
    return distance <= config.fp_projzerotol


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
