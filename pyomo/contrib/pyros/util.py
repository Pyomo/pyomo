'''
Utility functions for the PyROS solver
'''
import copy
from enum import Enum, auto
from pyomo.common.collections import ComponentSet
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (Constraint, Var, Param,
                             Objective, minimize, Expression,
                             ConcreteModel, maximize, Block)
from pyomo.core.base.var import IndexedVar
from pyomo.core.base.set_types import Reals
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr import value
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
from pyomo.core.expr.sympy_tools import sympyify_expression, sympy2pyomo_expression
from pyomo.common.dependencies import scipy as sp
import itertools as it
import timeit
from contextlib import contextmanager
import logging
from pprint import pprint

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

class grcsTerminationCondition(Enum):
    '''
    Enum class to describe termination conditions of the grcs algorithm
    robust_optimal: the grcs algorithm returned with a robust_optimal solution under normal conditions
    max_iter: the grcs algorithm could not identify a robust optimal solution within the specified max_iter.
              Consider increasing the max_iter config param.
    error: there was an error in the grcs solution procedure. Check the log file.
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
    nom_vals = list(p.value for p in config.uncertain_params)
    if not config.uncertainty_set.point_in_set(uncertain_params=config.uncertain_params, point=nom_vals):
        raise AttributeError("Nominal point for uncertain parameters must be in the uncertainty set.")

    # === Add nominal point to config
    config.nominal_uncertain_param_vals = nom_vals

    return


def add_bounds_for_uncertain_parameters(separation_model, config):
    '''
    This function solves a set of optimization problems to determine bounds on the uncertain parameters
    given the uncertainty set description. These bounds will be added as additional constraints to the uncertaint_set_constr
    constraint. Should only be called once set_as_constraint() has been called on the separation_model object.
    :param separation_model: the model on which to add the bounds
    :param config: solver config
    :return:
    '''
    # === Determine bounds on all uncertain params
    uncertain_param_bounds = []
    bounding_model = ConcreteModel()
    bounding_model.util = Block()
    bounding_model.util.uncertain_param_vars = IndexedVar(separation_model.util.uncertain_param_vars.index_set())
    for tup in separation_model.util.uncertain_param_vars.items():
        bounding_model.util.uncertain_param_vars[tup[0]].value = tup[1].value

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
        separation_model.util.uncertain_param_vars[idx].setlb(bound[0])
        separation_model.util.uncertain_param_vars[idx].setub(bound[1])

    return


def transform_to_standard_form(model):
    '''
    Make all inequality constraints of the form g(x) <= 0
    :param model: the optimization model
    :return: void
    '''
    for constraint in model.component_data_objects(Constraint, descend_into=True, active=True):
        if not constraint.equality:
            if constraint.lower is not None:
                temp = constraint
                model.del_component(constraint)
                model.add_component(temp.name, Constraint(expr= - (temp.body) <= - (temp.lower)))

    return


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
        raise ValueError("First-stage variables (first_stage_variables) and "
                         "second-stage variables (second_stage_variables) cannot both be empty lists.")

    if ComponentSet(first_stage_variables) != ComponentSet(config.first_stage_variables):
        raise ValueError("First-stage variables in first_stage_variables must be members of the model object.")

    if ComponentSet(second_stage_variables) != ComponentSet(config.second_stage_variables):
        raise ValueError("Second-stage variables in second_stage_variables must be members of the model object.")

    if ComponentSet(uncertain_params) != ComponentSet(config.uncertain_params):
        raise ValueError("Uncertain parameters in uncertain_params must be members of the model object.")

    if not config.uncertainty_set:
        raise ValueError("An UncertaintySet object must be provided to the PyROS solver.")

    non_mutable_params = []
    for p in config.uncertain_params:
        if not (not p.is_constant() and p.is_fixed() and not p.is_potentially_variable()):
            non_mutable_params.append(p)
        if non_mutable_params:
            raise ValueError("Param objects which are uncertain must have attribute mutable=True. "
                             "Offending Params: %s" % [p.name for p in non_mutable_params])

    # === Ensure that if there is an uncertain_param in an equality constraint,
    #     there is at least 1 state or 1 second-stage var as well
    uncertain_param_set = ComponentSet(config.uncertain_params)
    first_stage_variable_set = ComponentSet(config.first_stage_variables)
    for c in model.component_data_objects(Constraint, active=True):
        if c.equality:
            vars_in_term = list(v for v in identify_variables(c.expr))
            uncertain_params = list(p for p in identify_mutable_parameters(c.expr)
                                   if p in uncertain_param_set)
            if len(uncertain_params) > 0:
                state_vars_in_expr = list(v for v in vars_in_term
                                          if v in first_stage_variable_set)
                second_stage_vars_in_expr = list(v for v in vars_in_term
                                                 if v not in first_stage_variable_set)
                if len(state_vars_in_expr) == 0 and len(second_stage_vars_in_expr) == 0:
                    raise AttributeError("PyROS assumption violated: if any uncertain parameters participate"
                                         "in an equality constraint, either a state or second-stage variable must"
                                         "also participate. Offending constraint: %s " % c.name)

    # === Solvers provided check
    if not config.local_solver or not config.global_solver:
        raise ValueError("User must designate both a local and global optimization solver via the local_solver"
                         " and global_solver options.")

    # === Degrees of freedom provided check
    if len(config.first_stage_variables) + len(config.second_stage_variables) == 0:
        raise ValueError("User must designate at least one first- and/or second-stage variable.")

    # === Uncertain params provided check
    if len(config.uncertain_params) == 0:
        raise ValueError("User must designate at least one uncertain parameter.")

    return


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
            model_data.working_model.add_component("decision_rule_var_" + str(i),
                                                   Var(initialize=value(second_stage_variables[i]),bounds=bounds,domain=Reals))#bounds=(second_stage_variables[i].lb, second_stage_variables[i].ub)))
            first_stage_variables.extend(getattr(model_data.working_model, "decision_rule_var_" + str(i)).values())
            decision_rule_vars.append(getattr(model_data.working_model, "decision_rule_var_" + str(i)))
    elif degree == 1:
        for i in range(len(second_stage_variables)):
            index_set = list(range(len(uncertain_params) + 1))
            model_data.working_model.add_component("decision_rule_var_" + str(i),
                    Var(index_set,
                        initialize=0,
                        bounds=bounds,
                        domain=Reals))#bounds=(second_stage_variables[i].lb, second_stage_variables[i].ub)))
            # === For affine drs, the [0]th constant term is initialized to the control variable values, all other terms are initialized to 0
            getattr(model_data.working_model, "decision_rule_var_" + str(i))[0].value = value(second_stage_variables[i])
            first_stage_variables.extend(list(getattr(model_data.working_model, "decision_rule_var_" + str(i)).values()))
            decision_rule_vars.append(getattr(model_data.working_model, "decision_rule_var_" + str(i)))
    elif degree == 2 or degree == 3 or degree == 4:
        for i in range(len(second_stage_variables)):
            num_vars = int(sp.special.comb(N=len(uncertain_params) + degree, k=degree))
            dict_init = {}
            for r in range(num_vars):
                if r == 0:
                    dict_init.update({r: value(second_stage_variables[i])})
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
    return


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
            all_powers.append(list(partition_powers(n, len(uncertain_params))))
        for i in range(len(second_stage_variables)):
            Z = list(z for z in getattr(model_data.working_model, "decision_rule_var_" + str(i)).values())
            e = Z.pop(0)
            for degree_param_powers in all_powers:
                degree_param_powers = degree_param_powers[::-1]
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
    return


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
                             "Please specity your objective function as minimization.")
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
    m.first_stage_objective = Expression(expr=first_stage_cost_expr + const_obj_expr)
    m.second_stage_objective = Expression(expr=second_stage_cost_expr)
    return


def load_final_solution(model_data, master_soln):
    '''
    load the final solution into the original model object
    :param model_data: model data container object
    :param master_soln: results data container object returned to user
    :return:
    '''
    model = model_data.original_model
    soln = master_soln.nominal_block

    src_vars = getattr(model, 'tmp_var_list')
    local_vars = getattr(soln, 'tmp_var_list')
    varMap = list(zip(src_vars, local_vars))

    for src, local in varMap:
        src.value = local.value

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
            return (False, grcsTerminationCondition.robust_infeasible)
        elif termination_condition in try_backups:
            return (True, None)
        else:
            raise NotImplementedError("This solver return termination condition (%s) "
                                      "is currently not supported by PyROS." % termination_condition)
    else:
        if termination_condition in globally_acceptable:
            return (False, None)
        elif termination_condition in robust_infeasible:
            return (False, grcsTerminationCondition.robust_infeasible)
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
                    "PyROS is still under development. This version is a beta release.\n"
                    "Please provide feedback and/or report any issues by opening a Pyomo ticket.\n"
                    "===========================================================================================\n")
    # === ALL LOGGER RETURN MESSAGES
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
