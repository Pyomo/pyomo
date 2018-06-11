"""Utility functions and classes for the GDPopt solver."""
from __future__ import division

import logging
from math import fabs, floor, log

from pyomo.core import (Any, Binary, Block, Constraint, NonNegativeReals,
                        Objective, Reals, Var, minimize, value)
from pyomo.core.expr import current as EXPR
from pyomo.core.kernel import ComponentSet
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory
from pyomo.opt.results import ProblemSense, SolverResults


class _DoNothing(object):
    """Do nothing, literally.

    This class is used in situations of "do something if attribute exists."
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def __getattr__(self, attr):
        def _do_nothing(*args, **kwargs):
            pass
        return _do_nothing


class GDPoptSolveData(object):
    """Data container to hold solve-instance data.

    Key attributes:
        - original_model: the original model that the user gave us to solve
        - working_model: the original model after preprocessing
        - linear_GDP: the linear-discrete master problem

    """
    pass


def model_is_valid(solve_data, config):
    """Validate that the model is solveable by GDPopt.

    Also preforms some preprocessing such as moving the objective to the
    constraints.

    """
    m = solve_data.working_model
    GDPopt = m.GDPopt_utils

    # Handle LP/NLP being passed to the solver
    prob = solve_data.results.problem
    if (prob.number_of_binary_variables == 0 and
        prob.number_of_integer_variables == 0 and
            prob.number_of_disjunctions == 0):
        config.logger.info('Problem has no discrete decisions.')
        if len(GDPopt.working_nonlinear_constraints) > 0:
            config.logger.info(
                "Your model is an NLP (nonlinear program). "
                "Using NLP solver %s to solve." % config.nlp)
            SolverFactory(config.nlp).solve(
                solve_data.original_model, **config.nlp_options)
            return False
        else:
            config.logger.info(
                "Your model is an LP (linear program). "
                "Using LP solver %s to solve." % config.mip)
            SolverFactory(config.mip).solve(
                solve_data.original_model, **config.mip_options)
            return False

    # Handle missing or multiple objectives
    objs = list(m.component_data_objects(
        ctype=Objective, active=True, descend_into=True))
    num_objs = len(objs)
    solve_data.results.problem.number_of_objectives = num_objs
    if num_objs == 0:
        config.logger.warning(
            'Model has no active objectives. Adding dummy objective.')
        GDPopt.dummy_objective = Objective(expr=1)
        main_obj = GDPopt.dummy_objective
    elif num_objs > 1:
        raise ValueError('Model has multiple active objectives.')
    else:
        main_obj = objs[0]
    solve_data.working_objective_expr = main_obj.expr

    # Move the objective to the constraints

    # TODO only move the objective if nonlinear?
    GDPopt.objective_value = Var(domain=Reals, initialize=0)
    solve_data.objective_sense = main_obj.sense
    if main_obj.sense == minimize:
        GDPopt.objective_expr = Constraint(
            expr=GDPopt.objective_value >= main_obj.expr)
        solve_data.results.problem.sense = ProblemSense.minimize
    else:
        GDPopt.objective_expr = Constraint(
            expr=GDPopt.objective_value <= main_obj.expr)
        solve_data.results.problem.sense = ProblemSense.maximize
    main_obj.deactivate()
    GDPopt.objective = Objective(
        expr=GDPopt.objective_value, sense=main_obj.sense)

    # TODO if any continuous variables are multipled with binary ones, need
    # to do some kind of transformation (Glover?) or throw an error message
    return True


def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)


def copy_var_list_values(from_list, to_list, config, skip_stale=False):
    """Copy variable values from one list to another."""
    for v_from, v_to in zip(from_list, to_list):
        if skip_stale and v_from.stale:
            continue  # Skip stale variable values.
        try:
            v_to.set_value(value(v_from, exception=False))
            if skip_stale:
                v_to.stale = False
        except ValueError as err:
            if 'is not in domain Binary' in err.message:
                # Check to see if this is just a tolerance issue
                v_from_val = value(v_from, exception=False)
                if (fabs(v_from_val - 1) <= config.integer_tolerance or
                        fabs(v_from_val) <= config.integer_tolerance):
                    v_to.set_value(round(v_from_val))
                else:
                    raise


def is_feasible(model, config):
    """Checks to see if the algebraic model is feasible in its current state.

    Checks variable bounds and active constraints. Not for use with
    untransformed GDP models.

    """
    disj = next(model.component_data_objects(
        ctype=Disjunct, active=True), None)
    if disj is not None:
        raise NotImplementedError(
            "Found active disjunct %s. "
            "This function is not intended to check "
            "feasibility of disjunctive models, "
            "only transformed subproblems." % disj.name)

    config.logger.debug('Checking if model is feasible.')
    for constr in model.component_data_objects(
            ctype=Constraint, active=True, descend_into=True):
        # Check constraint lower bound
        if (constr.lower is not None and (
            value(constr.lower) - value(constr.body)
            >= config.constraint_tolerance
        )):
            config.logger.info('%s: body %s < LB %s' % (
                constr.name, value(constr.body), value(constr.lower)))
            return False
        # check constraint upper bound
        if (constr.upper is not None and (
            value(constr.body) - value(constr.upper)
            >= config.constraint_tolerance
        )):
            config.logger.info('%s: body %s > UB %s' % (
                constr.name, value(constr.body), value(constr.upper)))
            return False
    for var in model.component_data_objects(ctype=Var, descend_into=True):
        # Check variable lower bound
        if (var.has_lb() and
                value(var.lb) - value(var) >= config.variable_tolerance):
            config.logger.info('%s: %s < LB %s' % (
                var.name, value(var), value(var.lb)))
            return False
        # Check variable upper bound
        if (var.has_ub() and
                value(var) - value(var.ub) >= config.variable_tolerance):
            config.logger.info('%s: %s > UB %s' % (
                var.name, value(var), value(var.ub)))
            return False
    config.logger.info('Model is feasible.')
    return True


def clone_orig_model_with_lists(original_model):
    """Clones the original model to create a working model.

    Also attaches ordered lists of the variables, constraints, disjuncts, and
    disjunctions to the model so that they can be used for mapping back and
    forth.

    """
    build_ordered_component_lists(original_model, prefix='orig')
    return original_model.clone()


def build_ordered_component_lists(model, prefix='working'):
    """Define lists used for future data transfer."""
    GDPopt = model.GDPopt_utils
    var_set = ComponentSet()
    setattr(
        GDPopt, '%s_constraints_list' % prefix, list(
            model.component_data_objects(
                ctype=Constraint, active=True,
                descend_into=(Block, Disjunct))))
    setattr(
        GDPopt, '%s_disjuncts_list' % prefix, list(
            model.component_data_objects(
                ctype=Disjunct, descend_into=(Block, Disjunct))))
    setattr(
        GDPopt, '%s_disjunctions_list' % prefix, list(
            model.component_data_objects(
                ctype=Disjunction, active=True,
                descend_into=(Disjunct, Block))))

    # Identify the non-fixed variables in (potentially) active constraints
    for constr in getattr(GDPopt, '%s_constraints_list' % prefix):
        for v in EXPR.identify_variables(constr.body, include_fixed=False):
            var_set.add(v)
    # Disjunct indicator variables might not appear in active constraints. In
    # fact, if we consider them Logical variables, they should not appear in
    # active algebraic constraints. For now, they need to be added to the
    # variable set.
    for disj in getattr(GDPopt, '%s_disjuncts_list' % prefix):
        var_set.add(disj.indicator_var)

    # We use component_data_objects rather than list(var_set) in order to
    # preserve a deterministic ordering.
    setattr(
        GDPopt, '%s_var_list' % prefix, list(
            v for v in model.component_data_objects(
                ctype=Var, descend_into=(Block, Disjunct))
            if v in var_set))
    setattr(
        GDPopt, '%s_nonlinear_constraints' % prefix, [
            v for v in getattr(GDPopt, '%s_constraints_list' % prefix)
            if v.body.polynomial_degree() not in (0, 1)])


def _record_problem_statistics(model, solve_data, config):
    # Create the solver results object
    res = solve_data.results = SolverResults()
    prob = res.problem
    GDPopt = model.GDPopt_utils
    origGDPopt = solve_data.original_model.GDPopt_utils
    res.problem.name = model.name
    res.problem.number_of_nonzeros = None  # TODO
    # TODO work on termination condition and message
    res.solver.termination_condition = None
    res.solver.message = None
    # TODO add some kind of timing
    res.solver.user_time = None
    res.solver.system_time = None
    res.solver.wallclock_time = None
    res.solver.termination_message = None

    # Classify the variables
    orig_binary = sum(1 for v in origGDPopt.orig_var_list if v.is_binary())
    orig_continuous = sum(
        1 for v in origGDPopt.orig_var_list if v.is_continuous())
    orig_integer = sum(1 for v in origGDPopt.orig_var_list if v.is_integer())
    now_binary = sum(1 for v in GDPopt.working_var_list if v.is_binary())
    now_continuous = sum(
        1 for v in GDPopt.working_var_list if v.is_continuous())
    now_integer = sum(1 for v in GDPopt.working_var_list if v.is_integer())
    assert now_integer == 0, "Unreformulated, unfixed integer variables found."

    # Get count of constraints and variables
    prob.number_of_constraints = len(GDPopt.orig_constraints_list)
    prob.number_of_disjunctions = len(GDPopt.orig_disjunctions_list)
    prob.number_of_variables = len(GDPopt.orig_var_list)
    prob.number_of_binary_variables = orig_binary
    prob.number_of_continuous_variables = orig_continuous
    prob.number_of_integer_variables = orig_integer

    config.logger.info(
        "Original model has %s constraints (%s nonlinear) "
        "and %s disjunctions, "
        "with %s variables, of which %s are binary, %s are integer, "
        "and %s are continuous." %
        (prob.number_of_constraints,
         len(GDPopt.orig_nonlinear_constraints),
         prob.number_of_disjunctions,
         prob.number_of_variables,
         orig_binary,
         orig_integer,
         orig_continuous))

    config.logger.info(
        "After preprocessing, model has %s constraints (%s nonlinear) "
        "and %s disjunctions, "
        "with %s variables, of which %s are binary and %s are continuous." %
        (len(GDPopt.working_constraints_list),
         len(GDPopt.working_nonlinear_constraints),
         len(GDPopt.working_disjunctions_list),
         len(GDPopt.working_var_list),
         now_binary,
         now_continuous))


def reformulate_integer_variables(model, config):
    integer_vars = list(
        v for v in model.component_data_objects(
            ctype=Var, descend_into=(Block, Disjunct))
        if v.is_integer() and not v.fixed)
    if len(integer_vars) == 0:
        return  # if no free integer variables, no reformulation needed.

    if config.reformulate_integer_vars_using is None:
        config.logger.warning(
            "Model contains unfixed integer variables. "
            "GDPopt will reformulate using base 2 binary variables "
            "by default. To specify a different method, see the "
            "reformulate_integer_vars_using configuration option.")
        config.reformulate_integer_vars_using = 'base2_binary'

    config.logger.info(
        "Reformulating integer variables using the %s strategy."
        % config.reformulate_integer_vars_using)

    # Set up reformulation block
    reform_block = model.GDPopt_utils.integer_reform = Block(
        doc="Holds variables and constraints for reformulating "
        "integer variables to binary variables.")
    reform_block.new_binary_var = Var(
        Any, domain=Binary, dense=False,
        doc="Binary variable with index (int_var.name, indx)")
    reform_block.integer_to_binary_constraint = Constraint(
        Any, doc="Equality constraints mapping the binary variable values "
        "to the integer variable value.")

    # check that variables are bounded and non-negative
    for int_var in integer_vars:
        if not (int_var.has_lb() and int_var.has_ub()):
            raise ValueError(
                "Integer variable %s is missing an "
                "upper or lower bound. LB: %s; UB: %s. "
                "GDPopt does not support unbounded integer variables."
                % (int_var.name, int_var.lb, int_var.ub))
        if int_var.lb < 0:
            raise ValueError(
                "Integer variable %s can be negative. "
                "GDPopt currently only supports positive integer "
                "variables." % (int_var.name)
            )
        # do the reformulation
        highest_power = floor(log(value(int_var.ub), 2))
        var_name = int_var.name
        reform_block.integer_to_binary_constraint.add(
            var_name, expr=int_var == sum(
                reform_block.new_binary_var[var_name, pwr] * (2 ** pwr)
                for pwr in range(0, int(highest_power) + 1)))
        int_var.domain = NonNegativeReals

    config.logger.info(
        "Reformulated %s integer variables using "
        "%s binary variables and %s constraints."
        % (len(integer_vars), len(reform_block.new_binary_var),
           len(reform_block.integer_to_binary_constraint)))


def validate_disjunctions(model, config):
    """Validate that the active disjunctions on the model are satisfied
    by the current disjunct indicator_var values."""
    active_disjunctions = model.component_data_objects(
        ctype=Disjunction, active=True, descend_into=(Block, Disjunct))
    for disjtn in active_disjunctions:
        sum_disj_vals = sum(disj.indicator_var.value
                            for disj in disjtn.disjuncts)
        if disjtn.xor and fabs(sum_disj_vals - 1) > config.integer_tolerance:
            raise ValueError(
                "Expected disjunct values to add up to 1 "
                "for XOR disjunction %s. "
                "Instead, values add up to %s." % (disjtn.name, sum_disj_vals))
        elif sum_disj_vals + config.integer_tolerance < 1:
            raise ValueError(
                "Expected disjunct values to add up to at least 1 for "
                "OR disjunction %s. "
                "Instead, values add up to %s." % (disjtn.name, sum_disj_vals))


def algorithm_is_making_progress(solve_data, config):
    """Make sure that the algorithm is making sufficient progress
    at each iteration to continue."""

    # TODO if backtracking is turned on, and algorithm visits the same point
    # twice without improvement in objective value, turn off backtracking.

    # TODO stop iterations if feasible solutions not progressing for a number
    # of iterations.

    # If the hybrid algorithm is not making progress, switch to OA.
    # required_feas_prog = 1E-6
    # if solve_data.working_model.GDPopt_utils.objective.sense == minimize:
    #     sign_adjust = 1
    # else:
    #     sign_adjust = -1

    # Maximum number of iterations in which feasible bound does not
    # improve before terminating algorithm
    # if (len(feas_prog_log) > config.algorithm_stall_after and
    #     (sign_adjust * (feas_prog_log[-1] + required_feas_prog)
    #      >= sign_adjust *
    #      feas_prog_log[-1 - config.algorithm_stall_after])):
    #     config.logger.info(
    #         'Feasible solutions not making enough progress '
    #         'for %s iterations. Algorithm stalled. Exiting.\n'
    #         'To continue, increase value of parameter '
    #         'algorithm_stall_after.'
    #         % (config.algorithm_stall_after,))
    #     return False

    return True


def algorithm_should_terminate(solve_data, config):
    """Check if the algorithm should terminate.

    Termination conditions based on solver options and progress.

    """
    # Check bound convergence
    if solve_data.LB + config.bound_tolerance >= solve_data.UB:
        config.logger.info(
            'GDPopt exiting on bound convergence. '
            'LB: %s + (tol %s) >= UB: %s' %
            (solve_data.LB, config.bound_tolerance,
             solve_data.UB))
        return True

    # Check iteration limit
    if solve_data.master_iteration >= config.iterlim:
        config.logger.info(
            'GDPopt unable to converge bounds '
            'after %s master iterations.'
            % (solve_data.master_iteration,))
        config.logger.info(
            'Final bound values: LB: %s  UB: %s'
            % (solve_data.LB, solve_data.UB))
        return True

    if not algorithm_is_making_progress(solve_data, config):
        config.logger.debug(
            'Algorithm is not making enough progress. '
            'Exiting iteration loop.')
        return True
    return False
