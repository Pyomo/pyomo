from collections import namedtuple
from heapq import heappush, heappop
from math import fabs

from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Block, Suffix, Constraint, ComponentMap, TransformationFactory
from pyomo.core.kernel.component_set import ComponentSet
from pyomo.gdp import Disjunct, Disjunction

_linear_degrees = {1, 0}


def _perform_branch_and_bound(solve_data):
    root_node = solve_data.working_model
    root_util_blk = root_node.GDPopt_utils
    config = solve_data.config
    obj_sign_correction = 1 if solve_data.objective_sense == minimize else -1

    # Map unfixed disjunct -> list of deactivated constraints
    root_util_blk.disjunct_to_nonlinear_constraints = ComponentMap()
    # Map relaxed disjunctions -> list of unfixed disjuncts
    root_util_blk.disjunction_to_unfixed_disjuncts = ComponentMap()

    # Preprocess the active disjunctions
    for disjunction in root_util_blk.disjunction_list:
        assert disjunction.active

        disjuncts_fixed_True = []
        disjuncts_fixed_False = []
        unfixed_disjuncts = []

        # categorize the disjuncts in the disjunction
        for disjunct in disjunction.disjuncts:
            if disjunct.indicator_var.fixed:
                if disjunct.indicator_var.value == 1:
                    disjuncts_fixed_True.append(disjunct)
                elif disjunct.indicator_var.value == 0:
                    disjuncts_fixed_False.append(disjunct)
                else:
                    pass  # raise error for fractional value?
            else:
                unfixed_disjuncts.append(disjunct)

        # update disjunct lists for predetermined disjunctions
        if len(disjuncts_fixed_False) == len(disjunction.disjuncts) - 1:
            # all but one disjunct in the disjunction is fixed to False.
            # Remaining one must be true. If not already fixed to True, do so.
            if not disjuncts_fixed_True:
                disjuncts_fixed_True = unfixed_disjuncts
                unfixed_disjuncts = []
                disjuncts_fixed_True[0].indicator_var.fix(1)
        elif disjuncts_fixed_True and disjunction.xor:
            assert len(disjuncts_fixed_True) == 1, "XOR (only one True) violated: %s" % disjunction.name
            disjuncts_fixed_False.extend(unfixed_disjuncts)
            unfixed_disjuncts = []

        # Make sure disjuncts fixed to False are properly deactivated.
        for disjunct in disjuncts_fixed_False:
            disjunct.deactivate()

        # Deactivate nonlinear constraints in unfixed disjuncts
        for disjunct in unfixed_disjuncts:
            nonlinear_constraints_in_disjunct = [
                constr for constr in disjunct.component_data_objects(Constraint, active=True)
                if constr.body.polynomial_degree() not in _linear_degrees]
            for constraint in nonlinear_constraints_in_disjunct:
                constraint.deactivate()
            if nonlinear_constraints_in_disjunct:
                # TODO might be worthwhile to log number of nonlinear constraints in each disjunction
                # for later branching purposes
                root_util_blk.disjunct_to_nonlinear_constraints[disjunct] = nonlinear_constraints_in_disjunct

        root_util_blk.disjunction_to_unfixed_disjuncts[disjunction] = unfixed_disjuncts
        pass

    # Add the BigM suffix if it does not already exist. Used later during nonlinear constraint activation.
    # TODO is this still necessary?
    if not hasattr(root_node, 'BigM'):
        root_node.BigM = Suffix()

    # Set up the priority queue
    queue = solve_data.bb_queue = []
    BBNodeData = namedtuple('BBNodeData', [
        'obj_lb',  # lower bound on objective value, sign corrected to minimize
        'obj_ub',  # upper bound on objective value, sign corrected to minimize
        'unbranched_disjunctions',  # number of unbranched disjunctions
        'node_count',  # cumulative node counter
    ])
    sort_tuple = BBNodeData(
        obj_lb=obj_sign_correction * float('-inf'),
        obj_ub=obj_sign_correction * float('inf'),
        unbranched_disjunctions=len(root_util_blk.disjunction_to_unfixed_disjuncts),
        node_count=0
    )
    heappush(queue, (sort_tuple, root_node))

    # Do the branch and bound
    while len(queue) > 0:
        # visit the top node on the heap
        node_data, node_model = queue[0]
        if node_data.obj_lb < node_data.obj_ub:  # TODO relax this with epsilon
            # Node has not been fully evaluated.
            _evaluate_node(node_data, node_model, solve_data)
        elif node_data.unbranched_disjunctions == 0:
            # We have reached a leaf node.
            original_model = solve_data.original_model
            for orig_var, val in zip(original_model.GDPopt_utils.variable_list,
                                     node_model.GDPopt_utils.variable_list):
                orig_var.value = val

            # TODO reevaluate these settings
            solve_data.results.problem.lower_bound = node_data.obj_lb
            solve_data.results.problem.upper_bound = node_data.obj_ub
            solve_data.results.solver.timing = solve_data.timing
            solve_data.results.solver.iterations = solve_data.explored_nodes
            # solve_data.results.solver.termination_condition = incumbent_results.solver.termination_condition
            return solve_data.results
        else:
            _branch_on_node()


def _evaluate_node(node_data, node_model, solve_data):
    solve_data.config.logger.info(
        "Exploring node with LB %.10g and %s inactive disjunctions." % (
            node_data.obj_lb, node_data.unbranched_disjunctions
        ))
    return _solve_rnGDP_subproblem(node_model, solve_data)


def _process_root_node(solve_data):
    config = solve_data.config
    root_node = solve_data.working_model
    obj_sign = solve_data.obj_sense_sign_factor  # 1 if minimize, -1 if maximize
    if config.check_sat and satisfiable(root_node, config.logger) is False:
        # Problem is not satisfiable. Problem is infeasible.
        config.logger.info("Root node is not satisfiable. Problem is infeasible.")
        obj_value = obj_sign * float('inf')
        return obj_value
    else:
        # solve the root node
        config.logger.info("Solving the root node.")
        obj_value, result, var_values = _solve_subproblem(root_node, solve_data)
        return obj_value


def _branch_on_node():
    pass


def _solve_rnGDP_subproblem(model, solve_data):
    config = solve_data.config
    TransformationFactory('gdp.bigm').apply_to(model)
    main_obj = next(subproblem.component_data_objects(Objective, active=True))
    obj_sign_correction = 1 if solve_data.objective_sense == minimize else -1

    try:
        result = SolverFactory(config.solver).solve(subproblem, **config.solver_args)
    except RuntimeError as e:
        config.logger.warning(
            "Solver encountered RuntimeError. Treating as infeasible. "
            "Msg: %s\n%s" % (str(e), traceback.format_exc()))
        var_values = [v.value for v in subproblem.GDPbb_utils.variable_list]
        return obj_sign * float('inf'), SolverResults(), var_values

    var_values = [v.value for v in subproblem.GDPbb_utils.variable_list]
    term_cond = result.solver.termination_condition
    if result.solver.status is SolverStatus.ok and any(
            term_cond == valid_cond for valid_cond in (tc.optimal, tc.locallyOptimal, tc.feasible)):
        return value(main_obj.expr), result, var_values
    elif term_cond == tc.unbounded:
        return obj_sign * float('-inf'), result, var_values
    elif term_cond == tc.infeasible:
        return obj_sign * float('inf'), result, var_values
    else:
        config.logger.warning("Unknown termination condition of %s" % term_cond)
        return obj_sign * float('inf'), result, var_values