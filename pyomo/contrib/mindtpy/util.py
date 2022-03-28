#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

"""Utility functions and classes for the MindtPy solver."""
from __future__ import division
import logging
from pyomo.common.collections import ComponentMap, Bunch
from pyomo.core import (Block, Constraint,
                        Objective, Reals, Suffix, Var, minimize, RangeSet, ConstraintList, TransformationFactory)
from pyomo.core.expr import differentiate
from pyomo.core.expr import current as EXPR
from pyomo.opt import SolverFactory, SolverResults
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
from pyomo.contrib.gdpopt.util import get_main_elapsed_time, time_code
from pyomo.core.expr.calculus.derivatives import differentiate
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.solvers.plugins.solvers.gurobi_direct import gurobipy
from pyomo.solvers.plugins.solvers.gurobi_persistent import GurobiPersistent
import math

pyomo_nlp = attempt_import('pyomo.contrib.pynumero.interfaces.pyomo_nlp')[0]
numpy = attempt_import('numpy')[0]


class MindtPySolveData(object):
    """Data container to hold solve-instance data.
    """
    pass


def model_is_valid(solve_data, config):
    """Determines whether the model is solvable by MindtPy.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.

    Returns
    -------
    bool
        True if model is solvable in MindtPy, False otherwise.
    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils

    # Handle LP/NLP being passed to the solver
    prob = solve_data.results.problem
    if len(MindtPy.discrete_variable_list) == 0:
        config.logger.info('Problem has no discrete decisions.')
        obj = next(m.component_data_objects(ctype=Objective, active=True))
        if (any(c.body.polynomial_degree() not in {1, 0} for c in MindtPy.constraint_list) or
                obj.expr.polynomial_degree() not in {1, 0}):
            config.logger.info(
                'Your model is a NLP (nonlinear program). '
                'Using NLP solver %s to solve.' % config.nlp_solver)
            nlpopt = SolverFactory(config.nlp_solver)
            set_solver_options(nlpopt, solve_data, config, solver_type='nlp')
            nlpopt.solve(solve_data.original_model,
                         tee=config.nlp_solver_tee, **config.nlp_solver_args)
            return False
        else:
            config.logger.info(
                'Your model is an LP (linear program). '
                'Using LP solver %s to solve.' % config.mip_solver)
            mainopt = SolverFactory(config.mip_solver)
            if isinstance(mainopt, PersistentSolver):
                mainopt.set_instance(solve_data.original_model)
            set_solver_options(mainopt, solve_data,
                               config, solver_type='mip')
            mainopt.solve(solve_data.original_model,
                          tee=config.mip_solver_tee, **config.mip_solver_args)
            return False

    if not hasattr(m, 'dual') and config.calculate_dual:  # Set up dual value reporting
        m.dual = Suffix(direction=Suffix.IMPORT)

    # TODO if any continuous variables are multiplied with binary ones,
    #  need to do some kind of transformation (Glover?) or throw an error message
    return True


def calc_jacobians(solve_data, config):
    """Generates a map of jacobians for the variables in the model.

    This function generates a map of jacobians corresponding to the variables in the
    model and adds this ComponentMap to solve_data.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    # Map nonlinear_constraint --> Map(
    #     variable --> jacobian of constraint wrt. variable)
    solve_data.jacobians = ComponentMap()
    if config.differentiate_mode == 'reverse_symbolic':
        mode = differentiate.Modes.reverse_symbolic
    elif config.differentiate_mode == 'sympy':
        mode = differentiate.Modes.sympy
    for c in solve_data.mip.MindtPy_utils.nonlinear_constraint_list:
        vars_in_constr = list(EXPR.identify_variables(c.body))
        jac_list = differentiate(
            c.body, wrt_list=vars_in_constr, mode=mode)
        solve_data.jacobians[c] = ComponentMap(
            (var, jac_wrt_var)
            for var, jac_wrt_var in zip(vars_in_constr, jac_list))


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
                    constr.body - constr.upper
                    <= MindtPy.feas_opt.slack_var[i])
            else:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.upper
                    <= MindtPy.feas_opt.slack_var)
        if constr.has_lb():
            if config.feasibility_norm in {'L1', 'L2'}:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.lower
                    >= -MindtPy.feas_opt.slack_var[i])
            else:
                MindtPy.feas_opt.feas_constraints.add(
                    constr.body - constr.lower
                    >= -MindtPy.feas_opt.slack_var)


def add_var_bound(solve_data, config):
    """This function will add bounds for variables in nonlinear constraints if they are not bounded.

    This is to avoid an unbounded main problem in the LP/NLP algorithm. Thus, the model will be 
    updated to include bounds for the unbounded variables in nonlinear constraints.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    """
    m = solve_data.working_model
    MindtPy = m.MindtPy_utils
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
    r"""This function generates objective (FP-NLP subproblem) for minimum euclidean distance to setpoint_model.

    L2 distance of (x,y) = \sqrt{\sum_i (x_i - y_i)^2}.

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
        The norm2 square objective function.
    """
    # skip objective_value variable and slack_var variables
    var_filter = (lambda v: v[1].is_integer()) if discrete_only \
        else (lambda v: 'MindtPy_utils.objective_value' not in v[1].name and
              'MindtPy_utils.feas_opt.slack_var' not in v[1].name)

    model_vars, setpoint_vars = zip(*filter(var_filter,
                                            zip(model.MindtPy_utils.variable_list,
                                                setpoint_model.MindtPy_utils.variable_list)))
    assert len(model_vars) == len(
        setpoint_vars), 'Trying to generate Squared Norm2 objective function for models with different number of variables'

    return Objective(expr=(
        sum([(model_var - setpoint_var.value)**2
             for (model_var, setpoint_var) in
             zip(model_vars, setpoint_vars)])))


def generate_norm1_objective_function(model, setpoint_model, discrete_only=False):
    r"""This function generates objective (PF-OA main problem) for minimum Norm1 distance to setpoint_model.

    Norm1 distance of (x,y) = \sum_i |x_i - y_i|.

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
        The norm1 objective function.
    """
    # skip objective_value variable and slack_var variables
    var_filter = (lambda v: v.is_integer()) if discrete_only \
        else (lambda v: 'MindtPy_utils.objective_value' not in v.name and
              'MindtPy_utils.feas_opt.slack_var' not in v.name)
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(
        filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(
        setpoint_vars), 'Trying to generate Norm1 objective function for models with different number of variables'
    model.MindtPy_utils.del_component('L1_obj')
    obj_blk = model.MindtPy_utils.L1_obj = Block()
    obj_blk.L1_obj_idx = RangeSet(len(model_vars))
    obj_blk.L1_obj_var = Var(
        obj_blk.L1_obj_idx, domain=Reals, bounds=(0, None))
    obj_blk.abs_reform = ConstraintList()
    for idx, v_model, v_setpoint in zip(obj_blk.L1_obj_idx, model_vars,
                                        setpoint_vars):
        obj_blk.abs_reform.add(
            expr=v_model - v_setpoint.value >= -obj_blk.L1_obj_var[idx])
        obj_blk.abs_reform.add(
            expr=v_model - v_setpoint.value <= obj_blk.L1_obj_var[idx])

    return Objective(expr=sum(obj_blk.L1_obj_var[idx] for idx in obj_blk.L1_obj_idx))


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
    var_filter = (lambda v: v.is_integer()) if discrete_only \
        else (lambda v: 'MindtPy_utils.objective_value' not in v.name and
              'MindtPy_utils.feas_opt.slack_var' not in v.name)
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(
        filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(
        setpoint_vars), 'Trying to generate Norm Infinity objective function for models with different number of variables'
    model.MindtPy_utils.del_component('L_infinity_obj')
    obj_blk = model.MindtPy_utils.L_infinity_obj = Block()
    obj_blk.L_infinity_obj_var = Var(domain=Reals, bounds=(0, None))
    obj_blk.abs_reform = ConstraintList()
    for v_model, v_setpoint in zip(model_vars,
                                   setpoint_vars):
        obj_blk.abs_reform.add(
            expr=v_model - v_setpoint.value >= -obj_blk.L_infinity_obj_var)
        obj_blk.abs_reform.add(
            expr=v_model - v_setpoint.value <= obj_blk.L_infinity_obj_var)

    return Objective(expr=obj_blk.L_infinity_obj_var)


def generate_lag_objective_function(model, setpoint_model, config, solve_data, discrete_only=False):
    """The function generates the second-order Taylor approximation of the Lagrangean.

    Parameters
    ----------
    model : Pyomo model
        The model that needs new objective function.
    setpoint_model : Pyomo model
        The model that provides the base point for us to calculate the distance.
    config : ConfigBlock
        The specific configurations for MindtPy.
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
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
    with time_code(solve_data.timing, 'PyomoNLP'):
        nlp = pyomo_nlp.PyomoNLP(temp_model)
        lam = [-temp_model.dual[constr] if abs(temp_model.dual[constr]) > config.zero_tolerance else 0
               for constr in nlp.get_pyomo_constraints()]
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
        first_order_term = sum(float(jac_lag[nlp.get_primal_indices([temp_var])[0]]) * (var - temp_var.value) for var,
                               temp_var in zip(model.MindtPy_utils.variable_list, temp_model.MindtPy_utils.variable_list) if temp_var.name in nlp_var)

        if config.add_regularization == 'grad_lag':
            return Objective(expr=first_order_term, sense=minimize)
        elif config.add_regularization in {'hess_lag', 'hess_only_lag'}:
            # Implementation 1
            hess_lag = nlp.evaluate_hessian_lag().toarray()
            hess_lag[abs(hess_lag) < config.zero_tolerance] = 0
            second_order_term = 0.5 * sum((var_i - temp_var_i.value) * float(hess_lag[nlp.get_primal_indices([temp_var_i])[0]][nlp.get_primal_indices([temp_var_j])[0]]) * (var_j - temp_var_j.value)
                                          for var_i, temp_var_i in zip(model.MindtPy_utils.variable_list, temp_model.MindtPy_utils.variable_list)
                                          for var_j, temp_var_j in zip(model.MindtPy_utils.variable_list, temp_model.MindtPy_utils.variable_list)
                                          if (temp_var_i.name in nlp_var and temp_var_j.name in nlp_var))
            if config.add_regularization == 'hess_lag':
                return Objective(expr=first_order_term + second_order_term, sense=minimize)
            elif config.add_regularization == 'hess_only_lag':
                return Objective(expr=second_order_term, sense=minimize)
        elif config.add_regularization == 'sqp_lag':
            var_filter = (lambda v: v[1].is_integer()) if discrete_only \
                else (lambda v: 'MindtPy_utils.objective_value' not in v[1].name and
                      'MindtPy_utils.feas_opt.slack_var' not in v[1].name)

            model_vars, setpoint_vars = zip(*filter(var_filter,
                                                    zip(model.MindtPy_utils.variable_list,
                                                        setpoint_model.MindtPy_utils.variable_list)))
            assert len(model_vars) == len(
                setpoint_vars), 'Trying to generate Squared Norm2 objective function for models with different number of variables'
            if config.sqp_lag_scaling_coef is None:
                rho = 1
            elif config.sqp_lag_scaling_coef == 'fixed':
                r = 1
                rho = numpy.linalg.norm(jac_lag/(2*r))
            elif config.sqp_lag_scaling_coef == 'variable_dependent':
                r = numpy.sqrt(
                    len(temp_model.MindtPy_utils.discrete_variable_list))
                rho = numpy.linalg.norm(jac_lag/(2*r))

            return Objective(expr=first_order_term + rho*sum([(model_var - setpoint_var.value)**2 for (model_var, setpoint_var) in zip(model_vars, setpoint_vars)]))


def generate_norm1_norm_constraint(model, setpoint_model, config, discrete_only=True):
    r"""This function generates constraint (PF-OA main problem) for minimum Norm1 distance to setpoint_model.

    Norm constraint is used to guarantees the monotonicity of the norm objective value sequence of all iterations
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
        Whether to only optimize on distance between the discrete variables, by default True.
    """
    var_filter = (lambda v: v.is_integer()) if discrete_only \
        else (lambda v: True)
    model_vars = list(filter(var_filter, model.MindtPy_utils.variable_list))
    setpoint_vars = list(
        filter(var_filter, setpoint_model.MindtPy_utils.variable_list))
    assert len(model_vars) == len(
        setpoint_vars), 'Trying to generate Norm1 norm constraint for models with different number of variables'
    norm_constraint_blk = model.MindtPy_utils.L1_norm_constraint = Block()
    norm_constraint_blk.L1_slack_idx = RangeSet(len(model_vars))
    norm_constraint_blk.L1_slack_var = Var(
        norm_constraint_blk.L1_slack_idx, domain=Reals, bounds=(0, None))
    norm_constraint_blk.abs_reform = ConstraintList()
    for idx, v_model, v_setpoint in zip(norm_constraint_blk.L1_slack_idx, model_vars,
                                        setpoint_vars):
        norm_constraint_blk.abs_reform.add(
            expr=v_model - v_setpoint.value >= -norm_constraint_blk.L1_slack_var[idx])
        norm_constraint_blk.abs_reform.add(
            expr=v_model - v_setpoint.value <= norm_constraint_blk.L1_slack_var[idx])
    rhs = config.fp_norm_constraint_coef * \
        sum(abs(v_model.value-v_setpoint.value)
            for v_model, v_setpoint in zip(model_vars, setpoint_vars))
    norm_constraint_blk.sum_slack = Constraint(
        expr=sum(norm_constraint_blk.L1_slack_var[idx] for idx in norm_constraint_blk.L1_slack_idx) <= rhs)


def set_solver_options(opt, solve_data, config, solver_type, regularization=False):
    """Set options for MIP/NLP solvers.

    Parameters
    ----------
    opt : SolverFactory
        The MIP/NLP solver.
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    config : ConfigBlock
        The specific configurations for MindtPy.
    solver_type : str
        The type of the solver, i.e. mip or nlp.
    regularization : bool, optional
        Whether the solver is used to solve the regularization problem, by default False.
    """
    # TODO: integrate nlp_args here
    # nlp_args = dict(config.nlp_solver_args)
    elapsed = get_main_elapsed_time(solve_data.timing)
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
    if solver_name in {'cplex', 'gurobi', 'gurobi_persistent'}:
        opt.options['timelimit'] = remaining
        opt.options['mipgap'] = config.mip_solver_mipgap
        if solver_name == 'gurobi_persistent' and config.single_tree:
            # PreCrush: Controls presolve reductions that affect user cuts
            # You should consider setting this parameter to 1 if you are using callbacks to add your own cuts.
            opt.set_gurobi_param('PreCrush', 1)
            opt.set_gurobi_param('LazyConstraints', 1)
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
        opt._solver_model.parameters.mip.tolerances.mipgap.set(
            config.mip_solver_mipgap)
        if regularization is True:
            if config.solution_limit is not None:
                opt._solver_model.parameters.mip.limits.solutions.set(
                    config.solution_limit)
            opt._solver_model.parameters.mip.strategy.presolvenode.set(3)
            if config.add_regularization in {'hess_lag', 'hess_only_lag'}:
                opt._solver_model.parameters.optimalitytarget.set(3)
    elif solver_name == 'glpk':
        opt.options['tmlim'] = remaining
        # TODO: mipgap does not work for glpk yet
        # opt.options['mipgap'] = config.mip_solver_mipgap
    elif solver_name == 'baron':
        opt.options['MaxTime'] = remaining
        opt.options['AbsConFeasTol'] = config.zero_tolerance
    elif solver_name == 'ipopt':
        opt.options['max_cpu_time'] = remaining
        opt.options['constr_viol_tol'] = config.zero_tolerance
    elif solver_name == 'gams':
        if solver_type == 'mip':
            opt.options['add_options'] = ['option optcr=%s;' % config.mip_solver_mipgap,
                                          'option reslim=%s;' % remaining]
        elif solver_type == 'nlp':
            opt.options['add_options'] = ['option reslim=%s;' % remaining]
            if config.nlp_solver_args.__contains__('solver'):
                if config.nlp_solver_args['solver'] in {'ipopt', 'ipopth', 'msnlp', 'conopt', 'baron'}:
                    if config.nlp_solver_args['solver'] == 'ipopt':
                        opt.options['add_options'].append(
                            '$onecho > ipopt.opt')
                        opt.options['add_options'].append(
                            'constr_viol_tol ' + str(config.zero_tolerance))
                    elif config.nlp_solver_args['solver'] == 'ipopth':
                        opt.options['add_options'].append(
                            '$onecho > ipopth.opt')
                        opt.options['add_options'].append(
                            'constr_viol_tol ' + str(config.zero_tolerance))
                        # TODO: Ipopt warmstart option
                        # opt.options['add_options'].append('warm_start_init_point       yes\n'
                        #                                   'warm_start_bound_push       1e-9\n'
                        #                                   'warm_start_bound_frac       1e-9\n'
                        #                                   'warm_start_slack_bound_frac 1e-9\n'
                        #                                   'warm_start_slack_bound_push 1e-9\n'
                        #                                   'warm_start_mult_bound_push  1e-9\n')
                    elif config.nlp_solver_args['solver'] == 'conopt':
                        opt.options['add_options'].append(
                            '$onecho > conopt.opt')
                        opt.options['add_options'].append(
                            'RTNWMA ' + str(config.zero_tolerance))
                    elif config.nlp_solver_args['solver'] == 'msnlp':
                        opt.options['add_options'].append(
                            '$onecho > msnlp.opt')
                        opt.options['add_options'].append(
                            'feasibility_tolerance ' + str(config.zero_tolerance))
                    elif config.nlp_solver_args['solver'] == 'baron':
                        opt.options['add_options'].append(
                            '$onecho > baron.opt')
                        opt.options['add_options'].append(
                            'AbsConFeasTol ' + str(config.zero_tolerance))
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


def set_up_solve_data(model, config):
    """Set up the solve data.

    Parameters
    ----------
    model : Pyomo model
        The original model to be solved in MindtPy.
    config : ConfigBlock
        The specific configurations for MindtPy.

    Returns
    -------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    """
    solve_data = MindtPySolveData()
    solve_data.results = SolverResults()
    solve_data.timing = Bunch()
    solve_data.curr_int_sol = []
    solve_data.should_terminate = False
    solve_data.integer_list = []

    # if the objective function is a constant, dual bound constraint is not added.
    obj = next(model.component_data_objects(ctype=Objective, active=True))
    if obj.expr.polynomial_degree() == 0:
        config.use_dual_bound = False

    if config.use_fbbt:
        fbbt(model)
        # TODO: logging_level is not logging.INFO here
        config.logger.info(
            'Use the fbbt to tighten the bounds of variables')

    solve_data.original_model = model
    solve_data.working_model = model.clone()

    # Set up iteration counters
    solve_data.nlp_iter = 0
    solve_data.mip_iter = 0
    solve_data.mip_subiter = 0
    solve_data.nlp_infeasible_counter = 0
    if config.init_strategy == 'FP':
        solve_data.fp_iter = 1

    # set up bounds
    if obj.sense == minimize:
        solve_data.primal_bound = float('inf')
        solve_data.dual_bound = float('-inf')
    else:
        solve_data.primal_bound = float('-inf')
        solve_data.dual_bound = float('inf')
    solve_data.primal_bound_progress = [solve_data.primal_bound]
    solve_data.dual_bound_progress = [solve_data.dual_bound]
    solve_data.primal_bound_progress_time = [0]
    solve_data.dual_bound_progress_time = [0]
    solve_data.abs_gap = float('inf')
    solve_data.rel_gap = float('inf')
    solve_data.log_formatter = ' {:>9}   {:>15}   {:>15g}   {:>12g}   {:>12g}   {:>7.2%}   {:>7.2f}'
    solve_data.fixed_nlp_log_formatter = '{:1}{:>9}   {:>15}   {:>15g}   {:>12g}   {:>12g}   {:>7.2%}   {:>7.2f}'
    solve_data.log_note_formatter = ' {:>9}   {:>15}   {:>15}'
    if config.add_regularization is not None:
        if config.add_regularization in {'level_L1', 'level_L_infinity', 'grad_lag'}:
            solve_data.regularization_mip_type = 'MILP'
        elif config.add_regularization in {'level_L2', 'hess_lag', 'hess_only_lag', 'sqp_lag'}:
            solve_data.regularization_mip_type = 'MIQP'

    if config.single_tree and (config.add_no_good_cuts or config.use_tabu_list):
        solve_data.stored_bound = {}
    if config.strategy == 'GOA' and (config.add_no_good_cuts or config.use_tabu_list):
        solve_data.num_no_good_cuts_added = {}

    # Flag indicating whether the solution improved in the past
    # iteration or not
    solve_data.primal_bound_improved = False
    solve_data.dual_bound_improved = False

    if config.nlp_solver == 'ipopt':
        if not hasattr(solve_data.working_model, 'ipopt_zL_out'):
            solve_data.working_model.ipopt_zL_out = Suffix(
                direction=Suffix.IMPORT)
        if not hasattr(solve_data.working_model, 'ipopt_zU_out'):
            solve_data.working_model.ipopt_zU_out = Suffix(
                direction=Suffix.IMPORT)

    return solve_data


def copy_var_list_values_from_solution_pool(from_list, to_list, config, solver_model, var_map, solution_name,
                                            ignore_integrality=False):
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
                    solution_name, var_map[v_from])
            elif config.mip_solver == 'gurobi_persistent':
                solver_model.setParam(
                    gurobipy.GRB.Param.SolutionNumber, solution_name)
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
                    abs(var_val - rounded_val) <= config.integer_tolerance):
                v_to.set_value(rounded_val, skip_validation=True)
            elif abs(var_val) <= config.zero_tolerance and 0 in v_to.domain:
                v_to.set_value(0, skip_validation=True)
            else:
                config.logger.error(
                    'Unknown validation domain error setting variable %s' %
                    (v_to.name,)
                )
                raise


class GurobiPersistent4MindtPy(GurobiPersistent):
    """ A new persistent interface to Gurobi.

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
            self._callback_func(self._pyomo_model, self,
                                where, self.solve_data, self.config)
        return f


def update_gap(solve_data):
    """Update the relative gap and the absolute gap.

    Parameters
    ----------
    solve_data : MindtPySolveData
        Data container that holds solve-instance data.
    """
    solve_data.abs_gap = abs(solve_data.primal_bound - solve_data.dual_bound)
    solve_data.rel_gap = solve_data.abs_gap / (abs(solve_data.primal_bound) + 1E-10)


def update_dual_bound(solve_data, bound_value):
    """Update the dual bound.

    Call after solving relaxed problem, including relaxed NLP and MIP master problem.
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
        solve_data.dual_bound_improved = solve_data.dual_bound > solve_data.dual_bound_progress[-1]
    else:
        solve_data.dual_bound = min(bound_value, solve_data.dual_bound)
        solve_data.dual_bound_improved = solve_data.dual_bound < solve_data.dual_bound_progress[-1]
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
        solve_data.primal_bound_improved = solve_data.primal_bound < solve_data.primal_bound_progress[-1]
    else:
        solve_data.primal_bound = max(bound_value, solve_data.primal_bound)
        solve_data.primal_bound_improved = solve_data.primal_bound > solve_data.primal_bound_progress[-1]
    solve_data.primal_bound_progress.append(solve_data.primal_bound)
    solve_data.primal_bound_progress_time.append(get_main_elapsed_time(solve_data.timing))
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
            dual_bound_progress[i] = dual_bound * (1 - config.initial_bound_coef * solve_data.objective_sense * math.copysign(1,dual_bound))
        else:
            break
    for i in range(len(dual_bound_progress)):
        if i == 0:
            dual_integral += abs(dual_bound_progress[i] - solve_data.dual_bound) * (solve_data.dual_bound_progress_time[i])
        else:
            dual_integral += abs(dual_bound_progress[i] - solve_data.dual_bound) * (solve_data.dual_bound_progress_time[i] - solve_data.dual_bound_progress_time[i-1])
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
            primal_bound_progress[i] = primal_bound * (1 + config.initial_bound_coef * solve_data.objective_sense * math.copysign(1,primal_bound))
        else:
            break
    for i in range(len(primal_bound_progress)):
        if i == 0:
            primal_integral += abs(primal_bound_progress[i] - solve_data.primal_bound) * (solve_data.primal_bound_progress_time[i])
        else:
            primal_integral += abs(primal_bound_progress[i] - solve_data.primal_bound) * (solve_data.primal_bound_progress_time[i] - solve_data.primal_bound_progress_time[i-1])

    config.logger.info(' {:<25}:   {:>7.4f} '.format('Primal integral', primal_integral))
    return primal_integral
