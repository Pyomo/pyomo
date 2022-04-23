'''
Utility functions for the PyROS solver
'''
import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (Constraint, Var, ConstraintList,
                             Objective, minimize, Expression,
                             ConcreteModel, maximize, Block, Param)
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr import current as EXPR
from pyomo.core.expr.numeric_expr import NPV_MaxExpression, NPV_MinExpression
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters, replace_expressions
from pyomo.core.expr.sympy_tools import sympyify_expression, sympy2pyomo_expression
from pyomo.common.dependencies import scipy as sp
from pyomo.core.expr.numvalue import native_types
from pyomo.util.vars_from_expressions import get_vars_from_components
import itertools as it
import timeit
from contextlib import contextmanager
import logging
from pprint import pprint
import math

# Tolerances used in the code
PARAM_IS_CERTAIN_REL_TOL = 1e-4
PARAM_IS_CERTAIN_ABS_TOL = 0
COEFF_MATCH_REL_TOL = 1e-6
COEFF_MATCH_ABS_TOL = 0

'''Code borrowed from gdpopt: time_code, get_main_ellapsed_time, a_logger.'''
@contextmanager
def time_code(timing_data_obj, code_block_name, is_main_timer=False):
    """Starts timer at entry, stores elapsed time at exit

    If `is_main_timer=True`, the start time is stored in the timing_data_obj,
    allowing calculation of total elapsed time 'on the fly' (e.g. to enforce
    a time limit) using `get_main_elapsed_time(timing_data_obj)`.
    """
    start_time = timeit.default_timer()
    if is_main_timer:
        timing_data_obj.main_timer_start_time = start_time
    yield
    elapsed_time = timeit.default_timer() - start_time
    prev_time = timing_data_obj.get(code_block_name, 0)
    timing_data_obj[code_block_name] = prev_time + elapsed_time


def get_main_elapsed_time(timing_data_obj):
    """Returns the time since entering the main `time_code` context"""
    current_time = timeit.default_timer()
    try:
        return current_time - timing_data_obj.main_timer_start_time
    except AttributeError as e:
        if 'main_timer_start_time' in str(e):
           raise AttributeError(
                "You need to be in a 'time_code' context to use `get_main_elapsed_time()`."
            )

def a_logger(str_or_logger):
    """Returns a logger when passed either a logger name or logger object."""
    if isinstance(str_or_logger, logging.Logger):
        return str_or_logger
    else:
        return logging.getLogger(str_or_logger)

def ValidEnum(enum_class):
    '''
    Python 3 dependent format string
    '''
    def fcn(obj):
        if obj not in enum_class:
            raise ValueError("Expected an {0} object, "
                             "instead recieved {1}".format(enum_class.__name__, obj.__class__.__name__))
        return obj
    return fcn

class pyrosTerminationCondition(Enum):
    '''
    Enum class to describe termination conditions of the grcs algorithm
    robust_optimal: The grcs algorithm returned with a robust_optimal solution under normal conditions
    robust_feasible: The grcs algorithm determined a proven robust feasible solution.
                     See documentation for the distinction between robust feasible and robust optimal.
    robust_infeasible: The grcs algorithm terminated with a proof of robust infeasibility.
    max_iter: The grcs algorithm could not identify a robust optimal solution within the specified max_iter.
              Consider increasing the max_iter config param.
    subsolver_error: There was an error in the user-specified sub-solvers used in the grcs solution procedure. Check the sub-solver log files.
    time_out: The grcs algorithm could not identify a robust optimal solution within the specified time_limit.
    '''
    robust_feasible = 0
    robust_optimal = 1
    robust_infeasible = 2
    max_iter = 3
    subsolver_error = 4
    time_out = 5


class SeparationStrategy(Enum):
    all_violations = auto()
    max_violation = auto()


class SolveMethod(Enum):
    local_solve = auto()
    global_solve = auto()


class ObjectiveType(Enum):
    worst_case = auto()
    nominal = auto()

def model_is_valid(model):
    '''
    Possibilities:
    Deterministic model has a single objective
    Deterministic model has no objective
    Deterministic model has multiple objectives
    :param model: the deterministic model
    :return: True if it satisfies certain properties, else False.
    '''
    objectives = list(model.component_data_objects(Objective))
    for o in objectives:
        o.deactivate()
    if len(objectives) == 1:
        '''
        Ensure objective is a minimization. If not, change the sense.
        '''
        obj = objectives[0]

        if obj.sense is not minimize:
            sympy_obj = sympyify_expression(-obj.expr)
            # Use sympy to distribute the negation so the method for determining first/second stage costs is valid
            min_obj = Objective(expr=sympy2pyomo_expression(sympy_obj[1].simplify(), sympy_obj[0]))
            model.del_component(obj)
            model.add_component(unique_component_name(model, obj.name+'_min'), min_obj)
        return True

    elif len(objectives) > 1:
        '''
        User should deactivate all Objectives in the model except the one represented by the output of 
        first_stage_objective + second_stage_objective
        '''
        return False
    else:
        '''
        No Objective objects provided as part of the model, please provide an Objective to your model so that
        PyROS can infer first- and second-stage objective.
        '''
        return False


def turn_bounds_to_constraints(variable, model, config=None):
    '''
    Turn the variable in question's "bounds" into direct inequality constraints on the model.
    :param variable: the variable with bounds to be turned to None and made into constraints.
    :param model: the model in which the variable resides
    :param config: solver config
    :return: the list of inequality constraints that are the bounds
    '''
    if variable.lb is not None:
        name = variable.name + "_lower_bound_con"
        model.add_component(name, Constraint(expr=-variable <= -variable.lb))
        variable.setlb(None)
    if variable.ub is not None:
        name = variable.name + "_upper_bound_con"
        model.add_component(name, Constraint(expr=variable <= variable.ub))
        variable.setub(None)
    return


def get_time_from_solver(results):
    '''
    Based on the solver used (GAMS or other pyomo solver) the time is named differently. This function gets the time
    based on which sub-solver type is used.
    :param results: the results returned from the solver
    :return: time
    '''
    if hasattr(results.solver, "name"):
        if type(results.solver.name) == str:
            if "GAMS" in results.solver.name:
                return results.solver.user_time
            else:
                raise ValueError("Accessing the time for this type of solver is not supported by get_time_from_solver.")
        else:
            return results.solver.time
    else:
        return results.solver.time


def validate_uncertainty_set(config):
    '''
    Confirm expression output from uncertainty set function references all q in q.
    Typecheck the uncertainty_set.q is Params referenced inside of m.
    Give warning that the nominal point (default value in the model) is not in the specified uncertainty set.
    :param config: solver config
    '''
    # === Check that q in UncertaintySet object constraint expression is referencing q in model.uncertain_params
    uncertain_params = config.uncertain_params

    # === Non-zero number of uncertain parameters
    if len(uncertain_params) == 0:
        raise AttributeError("Must provide uncertain params, uncertain_params list length is 0.")
    # === No duplicate parameters
    if len(uncertain_params) != len(ComponentSet(uncertain_params)):
        raise AttributeError("No duplicates allowed for uncertain param objects.")
    # === Ensure nominal point is in the set
    if not config.uncertainty_set.point_in_set(point=config.nominal_uncertain_param_vals):
        raise AttributeError("Nominal point for uncertain parameters must be in the uncertainty set.")
    # === Check set validity via boundedness and non-emptiness
    if not config.uncertainty_set.is_valid(config=config):
        raise AttributeError("Invalid uncertainty set detected. Check the uncertainty set object to "
                             "ensure non-emptiness and boundedness.")

    return


def add_bounds_for_uncertain_parameters(model, config):
    '''
    This function solves a set of optimization problems to determine bounds on the uncertain parameters
    given the uncertainty set description. These bounds will be added as additional constraints to the uncertainty_set_constr
    constraint. Should only be called once set_as_constraint() has been called on the separation_model object.
    :param separation_model: the model on which to add the bounds
    :param config: solver config
    :return:
    '''
    # === Determine bounds on all uncertain params
    uncertain_param_bounds = []
    bounding_model = ConcreteModel()
    bounding_model.util = Block()
    bounding_model.util.uncertain_param_vars = IndexedVar(model.util.uncertain_param_vars.index_set())
    for tup in model.util.uncertain_param_vars.items():
        bounding_model.util.uncertain_param_vars[tup[0]].set_value(
            tup[1].value, skip_validation=True)

    bounding_model.add_component("uncertainty_set_constraint",
                                 config.uncertainty_set.set_as_constraint(
                                     uncertain_params=bounding_model.util.uncertain_param_vars, model=bounding_model,
                                     config=config
                                 ))

    for idx, param in enumerate(list(bounding_model.util.uncertain_param_vars.values())):
        bounding_model.add_component("lb_obj_" + str(idx), Objective(expr=param, sense=minimize))
        bounding_model.add_component("ub_obj_" + str(idx), Objective(expr=param, sense=maximize))

    for o in bounding_model.component_data_objects(Objective):
        o.deactivate()

    for i in range(len(bounding_model.util.uncertain_param_vars)):
        bounds = []
        for limit in ("lb", "ub"):
            getattr(bounding_model, limit + "_obj_" + str(i)).activate()
            res = config.global_solver.solve(bounding_model, tee=False)
            bounds.append(bounding_model.util.uncertain_param_vars[i].value)
            getattr(bounding_model, limit + "_obj_" + str(i)).deactivate()
        uncertain_param_bounds.append(bounds)

    # === Add bounds as constraints to uncertainty_set_constraint ConstraintList
    for idx, bound in enumerate(uncertain_param_bounds):
        model.util.uncertain_param_vars[idx].setlb(bound[0])
        model.util.uncertain_param_vars[idx].setub(bound[1])

    return


def transform_to_standard_form(model):
    """
    Recast all model inequality constraints of the form `a <= g(v)` (`<= b`)
    to the 'standard' form `a - g(v) <= 0` (and `g(v) - b <= 0`),
    in which `v` denotes all model variables and `a` and `b` are
    contingent on model parameters.

    Parameters
    ----------
    model : ConcreteModel
        The model to search for constraints. This will descend into all
        active Blocks and sub-Blocks as well.

    Note
    ----
    If `a` and `b` are identical and the constraint is not classified as an
    equality (i.e. the `equality` attribute of the constraint object
    is `False`), then the constraint is recast to the equality `g(v) == a`.
    """
    # Note: because we will be adding / modifying the number of
    # constraints, we want to resolve the generator to a list before
    # starting.
    cons = list(model.component_data_objects(
        Constraint, descend_into=True, active=True))
    for con in cons:
        if not con.equality:
            has_lb = con.lower is not None
            has_ub = con.upper is not None

            if has_lb and has_ub:
                if con.lower is con.upper:
                    # recast as equality Constraint
                    con.set_value(con.lower == con.body)
                else:
                    # range inequality; split into two Constraints.
                    uniq_name = unique_component_name(model, con.name + '_lb')
                    model.add_component(
                        uniq_name,
                        Constraint(expr=con.lower - con.body <= 0)
                    )
                    con.set_value(con.body - con.upper <= 0)
            elif has_lb:
                # not in standard form; recast.
                con.set_value(con.lower - con.body <= 0)
            elif has_ub:
                # move upper bound to body.
                con.set_value(con.body - con.upper <= 0)
            else:
                # unbounded constraint: deactivate
                con.deactivate()


def get_vars_from_component(block, ctype):
    """Determine all variables used in active components within a block.

    Parameters
    ----------
    block: Block
        The block to search for components.  This is a recursive
        generator and will descend into any active sub-Blocks as well.
    ctype:  class
        The component type (typically either :py:class:`Constraint` or
        :py:class:`Objective` to search for).

    """

    return get_vars_from_components(block, ctype, active=True,
                                    descend_into=True)


def replace_uncertain_bounds_with_constraints(model, uncertain_params):
    """
    For variables of which the bounds are dependent on the parameters
    in the list `uncertain_params`, remove the bounds and add
    explicit variable bound inequality constraints.

    :param model: Model in which to make the bounds/constraint replacements
    :type model: class:`pyomo.core.base.PyomoModel.ConcreteModel`
    :param uncertain_params: List of uncertain model parameters
    :type uncertain_params: list
    """
    uncertain_param_set = ComponentSet(uncertain_params)

    # component for explicit inequality constraints
    uncertain_var_bound_constrs = ConstraintList()
    model.add_component(unique_component_name(model,
                                              'uncertain_var_bound_cons'),
                        uncertain_var_bound_constrs)

    # get all variables in active objective and constraint expression(s)
    vars_in_cons = ComponentSet(get_vars_from_component(model, Constraint))
    vars_in_obj = ComponentSet(get_vars_from_component(model, Objective))

    for v in vars_in_cons | vars_in_obj:
        # get mutable parameters in variable bounds expressions
        ub = v.upper
        mutable_params_ub = ComponentSet(identify_mutable_parameters(ub))
        lb = v.lower
        mutable_params_lb = ComponentSet(identify_mutable_parameters(lb))

        # add explicit inequality constraint(s), remove variable bound(s)
        if mutable_params_ub & uncertain_param_set:
            if type(ub) is NPV_MinExpression:
                upper_bounds = ub.args
            else:
                upper_bounds = (ub,)
            for u_bnd in upper_bounds:
                uncertain_var_bound_constrs.add(v - u_bnd <= 0)
            v.setub(None)
        if mutable_params_lb & uncertain_param_set:
            if type(ub) is NPV_MaxExpression:
                lower_bounds = lb.args
            else:
                lower_bounds = (lb,)
            for l_bnd in lower_bounds:
                uncertain_var_bound_constrs.add(l_bnd - v <= 0)
            v.setlb(None)


def validate_kwarg_inputs(model, config):
    '''
    Confirm kwarg inputs satisfy PyROS requirements.
    :param model: the deterministic model
    :param config: the config for this PyROS instance
    :return:
    '''

    # === Check if model is ConcreteModel object
    if not isinstance(model, ConcreteModel):
        raise ValueError("Model passed to PyROS solver must be a ConcreteModel object.")

    first_stage_variables = config.first_stage_variables
    second_stage_variables = config.second_stage_variables
    uncertain_params = config.uncertain_params

    if not config.first_stage_variables and not config.second_stage_variables:
        # Must have non-zero DOF
        raise ValueError("first_stage_variables and "
                         "second_stage_variables cannot both be empty lists.")

    if ComponentSet(first_stage_variables) != ComponentSet(config.first_stage_variables):
        raise ValueError("All elements in first_stage_variables must be Var members of the model object.")

    if ComponentSet(second_stage_variables) != ComponentSet(config.second_stage_variables):
        raise ValueError("All elements in second_stage_variables must be Var members of the model object.")

    if any(v in ComponentSet(second_stage_variables) for v in ComponentSet(first_stage_variables)):
        raise ValueError("No common elements allowed between first_stage_variables and second_stage_variables.")

    if ComponentSet(uncertain_params) != ComponentSet(config.uncertain_params):
        raise ValueError("uncertain_params must be mutable Param members of the model object.")

    if not config.uncertainty_set:
        raise ValueError("An UncertaintySet object must be provided to the PyROS solver.")

    non_mutable_params = []
    for p in config.uncertain_params:
        if not (not p.is_constant() and p.is_fixed() and not p.is_potentially_variable()):
            non_mutable_params.append(p)
        if non_mutable_params:
            raise ValueError("Param objects which are uncertain must have attribute mutable=True. "
                             "Offending Params: %s" % [p.name for p in non_mutable_params])

    # === Solvers provided check
    if not config.local_solver or not config.global_solver:
        raise ValueError("User must designate both a local and global optimization solver via the local_solver"
                         " and global_solver options.")

    if config.bypass_local_separation and config.bypass_global_separation:
        raise ValueError("User cannot simultaneously enable options "
                         "'bypass_local_separation' and "
                         "'bypass_global_separation'.")

    # === Degrees of freedom provided check
    if len(config.first_stage_variables) + len(config.second_stage_variables) == 0:
        raise ValueError("User must designate at least one first- and/or second-stage variable.")

    # === Uncertain params provided check
    if len(config.uncertain_params) == 0:
        raise ValueError("User must designate at least one uncertain parameter.")


    return

def substitute_ssv_in_dr_constraints(model, constraint):
    '''
    Generate the standard_repn for the dr constraints. Generate new expression with replace_expression to ignore
    the ssv component.
    Then, replace_expression with substitution_map between ssv and the new expression.
    Deactivate or del_component the original dr equation.
    Then, return modified model and do coefficient matching as normal.
    :param model: the working_model
    :param constraint: an equality constraint from the working model identified to be of the form h(x,z,q) = 0.
    :return:
    '''
    dr_eqns = model.util.decision_rule_eqns
    fsv = ComponentSet(model.util.first_stage_variables)
    if not hasattr(model, "dr_substituted_constraints"):
        model.dr_substituted_constraints = ConstraintList()
    for eqn in dr_eqns:
        repn = generate_standard_repn(eqn.body, compute_values=False)
        new_expression = 0
        map_linear_coeff_to_var = [x for x in zip(repn.linear_coefs, repn.linear_vars) if x[1] in ComponentSet(fsv)]
        map_quad_coeff_to_var = [x for x in zip(repn.quadratic_coefs, repn.quadratic_vars) if x[1] in ComponentSet(fsv)]
        if repn.linear_coefs:
            for coeff, var in map_linear_coeff_to_var:
                new_expression += coeff * var
        if repn.quadratic_coefs:
            for coeff, var in map_quad_coeff_to_var:
                new_expression += coeff * var[0] * var[1] # var here is a 2-tuple

        model.no_ssv_dr_expr = Expression(expr=new_expression)
        substitution_map = {}
        substitution_map[id(repn.linear_vars[-1])] = model.no_ssv_dr_expr.expr

    model.dr_substituted_constraints.add(
            replace_expressions(expr=constraint.lower,
                                     substitution_map=substitution_map) ==
            replace_expressions(expr=constraint.body,
                                     substitution_map=substitution_map))

    # === Delete the original constraint
    model.del_component(constraint.name)
    model.del_component("no_ssv_dr_expr")

    return model.dr_substituted_constraints[max(model.dr_substituted_constraints.keys())]

def is_certain_parameter(uncertain_param_index, config):
    '''
    If an uncertain parameter's inferred LB and UB are within a relative tolerance,
    then the parameter is considered certain.
    :param uncertain_param_index: index of the parameter in the config.uncertain_params list
    :param config: solver config
    :return: True if param is effectively "certain," else return False
    '''
    if config.uncertainty_set.parameter_bounds:
        param_bounds = config.uncertainty_set.parameter_bounds[uncertain_param_index]
        return math.isclose(a=param_bounds[0], b=param_bounds[1],
                            rel_tol=PARAM_IS_CERTAIN_REL_TOL, abs_tol=PARAM_IS_CERTAIN_ABS_TOL)
    else:
        return False # cannot be determined without bounds

def coefficient_matching(model, constraint, uncertain_params, config):
    '''
    :param model: master problem model
    :param constraint: the constraint from the master problem model
    :param uncertain_params: the list of uncertain parameters
    :param first_stage_variables: the list of effective first-stage variables (includes ssv if decision_rule_order = 0)
    :return: True if the coefficient matching was successful, False if its proven robust_infeasible due to
             constraints of the form 1 == 0
    '''
    # === Returned flags
    successful_matching = True
    robust_infeasible = False

    # === Efficiency for q_LB = q_UB
    actual_uncertain_params = []

    for i in range(len(uncertain_params)):
        if not is_certain_parameter(uncertain_param_index=i, config=config):
            actual_uncertain_params.append(uncertain_params[i])

    # === Add coefficient matching constraint list
    if not hasattr(model, "coefficient_matching_constraints"):
        model.coefficient_matching_constraints = ConstraintList()
    if not hasattr(model, "swapped_constraints"):
        model.swapped_constraints = ConstraintList()

    variables_in_constraint = ComponentSet(identify_variables(constraint.expr))
    params_in_constraint = ComponentSet(identify_mutable_parameters(constraint.expr))
    first_stage_variables = model.util.first_stage_variables
    second_stage_variables = model.util.second_stage_variables

    # === Determine if we need to do DR expression/ssv substitution to
    #     make h(x,z,q) == 0 into h(x,d,q) == 0 (which is just h(x,q) == 0)
    if all(v in ComponentSet(first_stage_variables) for v in variables_in_constraint) and \
            any(q in ComponentSet(actual_uncertain_params) for q in params_in_constraint):
        # h(x, q) == 0
        pass
    elif all(v in ComponentSet(first_stage_variables + second_stage_variables) for v in variables_in_constraint) and \
            any(q in ComponentSet(actual_uncertain_params) for q in params_in_constraint):
        constraint = substitute_ssv_in_dr_constraints(model=model, constraint=constraint)
        variables_in_constraint = ComponentSet(identify_variables(constraint.expr))
        params_in_constraint = ComponentSet(identify_mutable_parameters(constraint.expr))
    else:
        pass

    if all(v in ComponentSet(first_stage_variables) for v in variables_in_constraint) and \
            any(q in ComponentSet(actual_uncertain_params) for q in params_in_constraint):

        # Swap param objects for variable objects in this constraint
        model.param_set = []
        for i in range(len(list(variables_in_constraint))):
            # Initialize Params to non-zero value due to standard_repn bug
            model.add_component("p_%s" % i, Param(initialize=1, mutable=True))
            model.param_set.append(getattr(model, "p_%s" % i))

        model.variable_set = []
        for i in range(len(list(actual_uncertain_params))):
            model.add_component("x_%s" % i, Var(initialize=1))
            model.variable_set.append(getattr(model, "x_%s" % i))

        original_var_to_param_map = list(zip(list(variables_in_constraint), model.param_set))
        original_param_to_vap_map = list(zip(list(actual_uncertain_params), model.variable_set))

        var_to_param_substitution_map_forward = {}
        # Separation problem initialized to nominal uncertain parameter values
        for var, param in original_var_to_param_map:
            var_to_param_substitution_map_forward[id(var)] = param

        param_to_var_substitution_map_forward = {}
        # Separation problem initialized to nominal uncertain parameter values
        for param, var in original_param_to_vap_map:
            param_to_var_substitution_map_forward[id(param)] = var

        var_to_param_substitution_map_reverse = {}
        # Separation problem initialized to nominal uncertain parameter values
        for var, param in original_var_to_param_map:
            var_to_param_substitution_map_reverse[id(param)] = var

        param_to_var_substitution_map_reverse = {}
        # Separation problem initialized to nominal uncertain parameter values
        for param, var in original_param_to_vap_map:
            param_to_var_substitution_map_reverse[id(var)] = param

        model.swapped_constraints.add(
            replace_expressions(
                expr=replace_expressions(expr=constraint.lower,
                                         substitution_map=param_to_var_substitution_map_forward),
                substitution_map=var_to_param_substitution_map_forward) ==
            replace_expressions(
                expr=replace_expressions(expr=constraint.body,
                                         substitution_map=param_to_var_substitution_map_forward),
                substitution_map=var_to_param_substitution_map_forward))

        swapped = model.swapped_constraints[max(model.swapped_constraints.keys())]

        val = generate_standard_repn(swapped.body, compute_values=False)

        if val.constant is not None:
            if type(val.constant) not in native_types:
                temp_expr = replace_expressions(val.constant, substitution_map=var_to_param_substitution_map_reverse)
                if temp_expr.is_potentially_variable():
                    model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                elif math.isclose(value(temp_expr), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
            elif math.isclose(value(val.constant), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                pass
            else:
                successful_matching = False
                robust_infeasible = True
        if val.linear_coefs is not None:
            for coeff in val.linear_coefs:
                if type(coeff) not in native_types:
                    temp_expr = replace_expressions(coeff, substitution_map=var_to_param_substitution_map_reverse)
                    if temp_expr.is_potentially_variable():
                        model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                    elif math.isclose(value(temp_expr), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                        pass
                    else:
                        successful_matching = False
                        robust_infeasible = True
                elif math.isclose(value(coeff), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
        if val.quadratic_coefs:
            for coeff in val.quadratic_coefs:
                if type(coeff) not in native_types:
                    temp_expr = replace_expressions(coeff, substitution_map=var_to_param_substitution_map_reverse)
                    if temp_expr.is_potentially_variable():
                        model.coefficient_matching_constraints.add(expr=temp_expr == 0)
                    elif math.isclose(value(temp_expr), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                        pass
                    else:
                        successful_matching = False
                        robust_infeasible = True
                elif math.isclose(value(coeff), 0, rel_tol=COEFF_MATCH_REL_TOL, abs_tol=COEFF_MATCH_ABS_TOL):
                    pass
                else:
                    successful_matching = False
                    robust_infeasible = True
        if val.nonlinear_expr is not None:
            successful_matching = False
            robust_infeasible = False

        if successful_matching:
            model.util.h_x_q_constraints.add(constraint)

    for i in range(len(list(variables_in_constraint))):
        model.del_component("p_%s" % i)

    for i in range(len(list(params_in_constraint))):
        model.del_component("x_%s" % i)

    model.del_component("swapped_constraints")
    model.del_component("swapped_constraints_index")

    return successful_matching, robust_infeasible


def selective_clone(block, first_stage_vars):
    """
    Clone everything in a base_model except for the first-stage variables
    :param block: the block of the model to be clones
    :param first_stage_vars: the variables which should not be cloned
    :return:
    """
    memo = {
        '__block_scope__': {id(block): True, id(None): False}
    }
    for v in first_stage_vars:
        memo[id(v)] = v
    new_block = copy.deepcopy(block, memo)
    new_block._parent = None

    return new_block


def add_decision_rule_variables(model_data, config):
    '''
    Function to add decision rule (DR) variables to the working model. DR variables become first-stage design
    variables which do not get copied at each iteration. Currently support static_approx (no DR), affine DR,
    and quadratic DR.
    :param model_data: the data container for the working model
    :param config: the config block
    :return:
    '''
    second_stage_variables = model_data.working_model.util.second_stage_variables
    first_stage_variables = model_data.working_model.util.first_stage_variables
    uncertain_params = model_data.working_model.util.uncertain_params
    decision_rule_vars = []
    degree = config.decision_rule_order
    bounds = (None, None)
    if degree == 0:
        for i in range(len(second_stage_variables)):
            model_data.working_model.add_component(
                    "decision_rule_var_" + str(i),
                    Var(initialize=value(second_stage_variables[i], exception=False),
                        bounds=bounds,domain=Reals)
            )
            first_stage_variables.extend(getattr(model_data.working_model, "decision_rule_var_" + str(i)).values())
            decision_rule_vars.append(getattr(model_data.working_model, "decision_rule_var_" + str(i)))
    elif degree == 1:
        for i in range(len(second_stage_variables)):
            index_set = list(range(len(uncertain_params) + 1))
            model_data.working_model.add_component("decision_rule_var_" + str(i),
                    Var(index_set,
                        initialize=0,
                        bounds=bounds,
                        domain=Reals))
            # === For affine drs, the [0]th constant term is initialized to the control variable values, all other terms are initialized to 0
            getattr(model_data.working_model, "decision_rule_var_" + str(i))[0].set_value(value(second_stage_variables[i], exception=False), skip_validation=True)
            first_stage_variables.extend(list(getattr(model_data.working_model, "decision_rule_var_" + str(i)).values()))
            decision_rule_vars.append(getattr(model_data.working_model, "decision_rule_var_" + str(i)))
    elif degree == 2 or degree == 3 or degree == 4:
        for i in range(len(second_stage_variables)):
            num_vars = int(sp.special.comb(N=len(uncertain_params) + degree, k=degree))
            dict_init = {}
            for r in range(num_vars):
                if r == 0:
                    dict_init.update({r: value(second_stage_variables[i], exception=False)})
                else:
                    dict_init.update({r: 0})
            model_data.working_model.add_component("decision_rule_var_" + str(i),
                                                   Var(list(range(num_vars)), initialize=dict_init, bounds=bounds,
                                                       domain=Reals))
            first_stage_variables.extend(
                list(getattr(model_data.working_model, "decision_rule_var_" + str(i)).values()))
            decision_rule_vars.append(getattr(model_data.working_model, "decision_rule_var_" + str(i)))
    else:
        raise ValueError(
            "Decision rule order " + str(config.decision_rule_order) +
            " is not yet supported. PyROS supports polynomials of degree 0 (static approximation), 1, 2.")
    model_data.working_model.util.decision_rule_vars = decision_rule_vars


def partition_powers(n, v):
    """Partition a total degree n across v variables

    This is an implementation of the "stars and bars" algorithm from
    combinatorial mathematics.

    This partitions a "total integer degree" of n across v variables
    such that each variable gets an integer degree >= 0.  You can think
    of this as dividing a set of n+v things into v groupings, with the
    power for each v_i being 1 less than the number of things in the
    i'th group (because the v is part of the group).  It is therefore
    sufficient to just get the v-1 starting points chosen from a list of
    indices n+v long (the first starting point is fixed to be 0).

    """
    for starts in it.combinations(range(1, n + v), v - 1):
        # add the initial starting point to the beginning and the total
        # number of objects (degree counters and variables) to the end
        # of the list.  The degree for each variable is 1 less than the
        # difference of sequential starting points (to account for the
        # variable itself)
        starts = (0,) + starts + (n+v,)
        yield [starts[i+1] - starts[i] - 1 for i in range(v)]

def sort_partitioned_powers(powers_list):
    powers_list = sorted(powers_list, reverse=True)
    powers_list = sorted(powers_list, key=lambda elem: max(elem))
    return powers_list


def add_decision_rule_constraints(model_data, config):
    '''
    Function to add the defining Constraint relationships for the decision rules to the working model.
    :param model_data: model data container object
    :param config: the config object
    :return:
    '''

    second_stage_variables = model_data.working_model.util.second_stage_variables
    uncertain_params = model_data.working_model.util.uncertain_params
    decision_rule_eqns = []
    degree = config.decision_rule_order
    if degree == 0:
        for i in range(len(second_stage_variables)):
            model_data.working_model.add_component("decision_rule_eqn_" + str(i),
                    Constraint(expr=getattr(model_data.working_model, "decision_rule_var_" + str(i)) == second_stage_variables[i]))
            decision_rule_eqns.append(getattr(model_data.working_model, "decision_rule_eqn_" + str(i)))
    elif degree == 1:
        for i in range(len(second_stage_variables)):
            expr = 0
            for j in range(len(getattr(model_data.working_model, "decision_rule_var_" + str(i)))):
                if j == 0:
                    expr += getattr(model_data.working_model, "decision_rule_var_" + str(i))[j]
                else:
                    expr += getattr(model_data.working_model, "decision_rule_var_" + str(i))[j] * uncertain_params[j - 1]
            model_data.working_model.add_component("decision_rule_eqn_" + str(i), Constraint(expr= expr == second_stage_variables[i]))
            decision_rule_eqns.append(getattr(model_data.working_model, "decision_rule_eqn_" + str(i)))
    elif degree >= 2:
        # Using bars and stars groupings of variable powers, construct x1^a * .... * xn^b terms for all c <= a+...+b = degree
        all_powers = []
        for n in range(1, degree+1):
            all_powers.append(sort_partitioned_powers(list(partition_powers(n, len(uncertain_params)))))
        for i in range(len(second_stage_variables)):
            Z = list(z for z in getattr(model_data.working_model, "decision_rule_var_" + str(i)).values())
            e = Z.pop(0)
            for degree_param_powers in all_powers:
                for param_powers in degree_param_powers:
                    product = 1
                    for idx, power in enumerate(param_powers):
                        if power == 0:
                            pass
                        else:
                            product = product * uncertain_params[idx]**power
                    e += Z.pop(0) * product
            model_data.working_model.add_component("decision_rule_eqn_" + str(i),
                                                       Constraint(expr=e == second_stage_variables[i]))
            decision_rule_eqns.append(getattr(model_data.working_model, "decision_rule_eqn_" + str(i)))
            if len(Z) != 0:
                raise RuntimeError("Construction of the decision rule functions did not work correctly! "
                                   "Did not use all coefficient terms.")
    model_data.working_model.util.decision_rule_eqns = decision_rule_eqns


def identify_objective_functions(model, config):
    '''
    Determine the objective first- and second-stage costs based on the user provided variable partition
    :param model: deterministic model
    :param config: config block
    :return:
    '''

    m = model
    obj = [o for o in model.component_data_objects(Objective)]
    if len(obj) > 1:
        raise AttributeError("Deterministic model must only have 1 active objective!")
    if obj[0].sense != minimize:
        raise AttributeError("PyROS requires deterministic models to have an objective function with  'sense'=minimization. "
                             "Please specify your objective function as minimization.")
    first_stage_terms = []
    second_stage_terms = []

    first_stage_cost_expr = 0
    second_stage_cost_expr = 0
    const_obj_expr = 0

    if isinstance(obj[0].expr, Var):
        obj_to_parse = [obj[0].expr]
    else:
        obj_to_parse = obj[0].expr.args
    first_stage_variable_set = ComponentSet(model.util.first_stage_variables)
    second_stage_variable_set = ComponentSet(model.util.second_stage_variables)
    for term in obj_to_parse:
        vars_in_term = list(v for v in identify_variables(term))

        first_stage_vars_in_term = list(v for v in vars_in_term if
                                        v in first_stage_variable_set)
        second_stage_vars_in_term = list(v for v in vars_in_term if
                                         v not in first_stage_variable_set)
        # By checking not in first_stage_variable_set, you pick up both ssv and state vars
        for v in first_stage_vars_in_term:
            if id(v) not in list(id(var) for var in first_stage_terms):
                first_stage_terms.append(v)
        for v in second_stage_vars_in_term:
            if id(v) not in list(id(var) for var in second_stage_terms):
                second_stage_terms.append(v)

        if first_stage_vars_in_term and second_stage_vars_in_term:
            second_stage_cost_expr += term
        elif first_stage_vars_in_term and not second_stage_vars_in_term:
            first_stage_cost_expr += term
        elif not first_stage_vars_in_term and second_stage_vars_in_term:
            second_stage_cost_expr += term
        elif not vars_in_term:
            const_obj_expr += term
    # convention to add constant objective term to first stage costs
    # IFF the const_obj_term does not contain an uncertain param! Else, it is second-stage cost
    mutable_params_in_const_term = identify_mutable_parameters(expr=const_obj_expr)
    if any(q in ComponentSet(model.util.uncertain_params) for q in mutable_params_in_const_term):
        m.first_stage_objective = Expression(expr=first_stage_cost_expr )
        m.second_stage_objective = Expression(expr=second_stage_cost_expr + const_obj_expr)
    else:
        m.first_stage_objective = Expression(expr=first_stage_cost_expr + const_obj_expr)
        m.second_stage_objective = Expression(expr=second_stage_cost_expr)
    return


def load_final_solution(model_data, master_soln, config):
    '''
    load the final solution into the original model object
    :param model_data: model data container object
    :param master_soln: results data container object returned to user
    :return:
    '''
    if config.objective_focus == ObjectiveType.nominal:
        model = model_data.original_model
        soln = master_soln.nominal_block
    elif config.objective_focus == ObjectiveType.worst_case:
        model = model_data.original_model
        indices = range(len(master_soln.master_model.scenarios))
        k = max(indices, key=lambda i: value(master_soln.master_model.scenarios[i, 0].first_stage_objective +
                                             master_soln.master_model.scenarios[i, 0].second_stage_objective))
        soln = master_soln.master_model.scenarios[k, 0]

    src_vars = getattr(model, 'tmp_var_list')
    local_vars = getattr(soln, 'tmp_var_list')
    varMap = list(zip(src_vars, local_vars))

    for src, local in varMap:
        src.set_value(local.value, skip_validation=True)

    return


def process_termination_condition_master_problem(config, results):
    '''
    :param config: pyros config
    :param results: solver results object
    :return: tuple (try_backups (True/False)
                  pyros_return_code (default NONE or robust_infeasible or subsolver_error))
    '''
    locally_acceptable = [tc.optimal, tc.locallyOptimal, tc.globallyOptimal]
    globally_acceptable = [tc.optimal, tc.globallyOptimal]
    robust_infeasible = [tc.infeasible]
    try_backups = [tc.feasible, tc.maxTimeLimit, tc.maxIterations, tc.maxEvaluations,
               tc.minStepLength, tc.minFunctionValue, tc.other, tc.solverFailure,
               tc.internalSolverError, tc.error,
               tc.unbounded, tc.infeasibleOrUnbounded, tc.invalidProblem, tc.intermediateNonInteger,
               tc.noSolution, tc.unknown]

    termination_condition = results.solver.termination_condition
    if config.solve_master_globally == False:
        if termination_condition in locally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError("This solver return termination condition (%s) "
                                      "is currently not supported by PyROS." % termination_condition)
    else:
        if termination_condition in globally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, pyrosTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError("This solver return termination condition (%s) "
                                      "is currently not supported by PyROS." % termination_condition)


def output_logger(config, **kwargs):
    '''
    All user returned messages (termination conditions, runtime errors) are here
    Includes when
    "sub-solver %s returned status infeasible..."
    :return:
    '''

    # === PREAMBLE + LICENSING
    # Version printing
    if "preamble" in kwargs:
        if kwargs["preamble"]:
            version = str(kwargs["version"])
            preamble = "===========================================================================================\n" \
                       "PyROS: Pyomo Robust Optimization Solver v.%s \n" \
                       "Developed by Natalie M. Isenberg (1), John D. Siirola (2), Chrysanthos E. Gounaris (1) \n" \
                       "(1) Carnegie Mellon University, Department of Chemical Engineering \n" \
                       "(2) Sandia National Laboratories, Center for Computing Research\n\n" \
                       "The developers gratefully acknowledge support from the U.S. Department of Energy's \n" \
                       "Institute for the Design of Advanced Energy Systems (IDAES) \n" \
                       "===========================================================================================" % version
            print(preamble)
    # === DISCLAIMER
    if "disclaimer" in kwargs:
        if kwargs["disclaimer"]:
           print("======================================== DISCLAIMER =======================================\n"
                    "PyROS is still under development. \n"
                    "Please provide feedback and/or report any issues by opening a Pyomo ticket.\n"
                    "===========================================================================================\n")
    # === ALL LOGGER RETURN MESSAGES
    if "bypass_global_separation" in kwargs:
        if kwargs["bypass_global_separation"]:
            config.progress_logger.info(
                    "NOTE: Option to bypass global separation was chosen. "
                    "Robust feasibility and optimality of the reported "
                    "solution are not guaranteed."
                    )
    if "robust_optimal" in kwargs:
        if kwargs["robust_optimal"]:
            config.progress_logger.info('Robust optimal solution identified. Exiting PyROS.')

    if "robust_feasible" in kwargs:
        if kwargs["robust_feasible"]:
            config.progress_logger.info('Robust feasible solution identified. Exiting PyROS.')

    if "robust_infeasible" in kwargs:
        if kwargs["robust_infeasible"]:
            config.progress_logger.info('Robust infeasible problem. Exiting PyROS.')


    if "time_out" in kwargs:
        if kwargs["time_out"]:
            config.progress_logger.info(
                'PyROS was unable to identify robust solution '
                'before exceeding time limit of %s seconds. '
                'Consider increasing the time limit via option time_limit.'
                 % config.time_limit)

    if "max_iter" in kwargs:
        if kwargs["max_iter"]:
            config.progress_logger.info(
                'PyROS was unable to identify robust solution '
                'within %s iterations of the GRCS algorithm. '
                'Consider increasing the iteration limit via option max_iter.'
                % config.max_iter)

    if "master_error" in kwargs:
        if kwargs["master_error"]:
            status_dict = kwargs["status_dict"]
            filename = kwargs["filename"]  # solver name to solver termination condition
            if kwargs["iteration"] == 0:
                raise AttributeError("User-supplied solver(s) could not solve the deterministic model. "
                                     "Returned termination conditions were: %s"
                                     "Please ensure deterministic model is solvable by at least one of the supplied solvers. "
                                     "Exiting PyROS." % pprint(status_dict, width=1))
            config.progress_logger.info(
                "User-supplied solver(s) could not solve the master model at iteration %s.\n"
                "Returned termination conditions were: %s\n"
                "For debugging, this problem has been written to a GAMS file titled %s. Exiting PyROS." % (kwargs["iteration"],
                                                                                                           pprint(status_dict),
                                                                                                           filename))
    if "separation_error" in kwargs:
        if kwargs["separation_error"]:
            status_dict = kwargs["status_dict"]
            filename = kwargs["filename"]
            iteration = kwargs["iteration"]
            obj = kwargs["objective"]
            config.progress_logger.info(
                "User-supplied solver(s) could not solve the separation problem at iteration %s under separation objective %s.\n"
                "Returned termination conditions were: %s\n"
                "For debugging, this problem has been written to a GAMS file titled %s. Exiting PyROS." % (iteration,
                                                                                                           obj,
                                                                                                           pprint(status_dict, width=1),
                                                                                                           filename))

    return
