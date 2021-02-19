#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import traceback
from collections import namedtuple
from heapq import heappush, heappop

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.util import copy_var_list_values, SuppressInfeasibleWarning, get_main_elapsed_time
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, Constraint, TransformationFactory
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition as tc

_linear_degrees = {1, 0}

# Data tuple for each node that also functions as the sort key.
# Therefore, ordering of the arguments below matters.
BBNodeData = namedtuple('BBNodeData', [
    'obj_lb',  # lower bound on objective value, sign corrected to minimize
    'obj_ub',  # upper bound on objective value, sign corrected to minimize
    'is_screened',  # True if the node has been screened; False if not.
    'is_evaluated',  # True if node has been evaluated; False if not.
    'num_unbranched_disjunctions',  # number of unbranched disjunctions
    'node_count',  # cumulative node counter
    'unbranched_disjunction_indices',  # list of unbranched disjunction indices
])


def _perform_branch_and_bound(solve_data):
    solve_data.explored_nodes = 0
    root_node = solve_data.working_model
    root_util_blk = root_node.GDPopt_utils
    config = solve_data.config

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
    solve_data.created_nodes = 0
    unbranched_disjunction_indices = [i for i, disjunction in enumerate(root_util_blk.disjunction_list)
                                      if disjunction in root_util_blk.disjunction_to_unfixed_disjuncts]
    sort_tuple = BBNodeData(
        obj_lb=float('-inf'),
        obj_ub=float('inf'),
        is_screened=False,
        is_evaluated=False,
        num_unbranched_disjunctions=len(unbranched_disjunction_indices),
        node_count=0,
        unbranched_disjunction_indices=unbranched_disjunction_indices,
    )
    heappush(queue, (sort_tuple, root_node))

    # Do the branch and bound
    while len(queue) > 0:
        # visit the top node on the heap
        # from pprint import pprint
        # pprint([(
        #     x[0].node_count, x[0].obj_lb, x[0].obj_ub, x[0].num_unbranched_disjunctions
        # ) for x in sorted(queue)])
        node_data, node_model = heappop(queue)
        config.logger.info("Nodes: %s LB %.10g Unbranched %s" % (
            solve_data.explored_nodes, node_data.obj_lb, node_data.num_unbranched_disjunctions))

        # Check time limit
        elapsed = get_main_elapsed_time(solve_data.timing)
        if elapsed >= config.time_limit:
            config.logger.info(
                'GDPopt-LBB unable to converge bounds '
                'before time limit of {} seconds. '
                'Elapsed: {} seconds'
                .format(config.time_limit, elapsed))
            no_feasible_soln = float('inf')
            solve_data.LB = node_data.obj_lb if solve_data.objective_sense == minimize else -no_feasible_soln
            solve_data.UB = no_feasible_soln if solve_data.objective_sense == minimize else -node_data.obj_lb
            config.logger.info(
                'Final bound values: LB: {}  UB: {}'.
                format(solve_data.LB, solve_data.UB))
            solve_data.results.solver.termination_condition = tc.maxTimeLimit
            return True

        # Handle current node
        if not node_data.is_screened:
            # Node has not been evaluated.
            solve_data.explored_nodes += 1
            new_node_data = _prescreen_node(node_data, node_model, solve_data)
            heappush(queue, (new_node_data, node_model))  # replace with updated node data
        elif node_data.obj_lb < node_data.obj_ub - config.bound_tolerance and not node_data.is_evaluated:
            # Node has not been fully evaluated.
            # Note: infeasible and unbounded nodes will skip this condition, because of strict inequality
            new_node_data = _evaluate_node(node_data, node_model, solve_data)
            heappush(queue, (new_node_data, node_model))  # replace with updated node data
        elif node_data.num_unbranched_disjunctions == 0 or node_data.obj_lb == float('inf'):
            # We have reached a leaf node, or the best available node is infeasible.
            original_model = solve_data.original_model
            copy_var_list_values(
                from_list=node_model.GDPopt_utils.variable_list,
                to_list=original_model.GDPopt_utils.variable_list,
                config=config,
            )

            solve_data.LB = node_data.obj_lb if solve_data.objective_sense == minimize else -node_data.obj_ub
            solve_data.UB = node_data.obj_ub if solve_data.objective_sense == minimize else -node_data.obj_lb
            solve_data.master_iteration = solve_data.explored_nodes
            if node_data.obj_lb == float('inf'):
                solve_data.results.solver.termination_condition = tc.infeasible
            elif node_data.obj_ub == float('-inf'):
                solve_data.results.solver.termination_condition = tc.unbounded
            else:
                solve_data.results.solver.termination_condition = tc.optimal
            return
        else:
            _branch_on_node(node_data, node_model, solve_data)


def _branch_on_node(node_data, node_model, solve_data):
    # Keeping the naive branch selection
    config = solve_data.config
    disjunction_to_branch_idx = node_data.unbranched_disjunction_indices[0]
    disjunction_to_branch = node_model.GDPopt_utils.disjunction_list[disjunction_to_branch_idx]
    num_unfixed_disjuncts = len(node_model.GDPopt_utils.disjunction_to_unfixed_disjuncts[disjunction_to_branch])
    config.logger.info("Branching on disjunction %s" % disjunction_to_branch.name)
    node_count = solve_data.created_nodes
    newly_created_nodes = 0

    for disjunct_index_to_fix_True in range(num_unfixed_disjuncts):
        # Create a new branch for each unfixed disjunct
        child_model = node_model.clone()
        child_disjunction_to_branch = child_model.GDPopt_utils.disjunction_list[disjunction_to_branch_idx]
        child_unfixed_disjuncts = child_model.GDPopt_utils.disjunction_to_unfixed_disjuncts[child_disjunction_to_branch]
        for idx, child_disjunct in enumerate(child_unfixed_disjuncts):
            if idx == disjunct_index_to_fix_True:
                child_disjunct.indicator_var.fix(1)
            else:
                child_disjunct.deactivate()
        if not child_disjunction_to_branch.xor:
            raise NotImplementedError("We still need to add support for non-XOR disjunctions.")
            # This requires adding all combinations of activation status among unfixed_disjuncts
        # Reactivate nonlinear constraints in the newly-fixed child disjunct
        fixed_True_disjunct = child_unfixed_disjuncts[disjunct_index_to_fix_True]
        for constr in child_model.GDPopt_utils.disjunct_to_nonlinear_constraints.get(fixed_True_disjunct, ()):
            constr.activate()
            child_model.BigM[constr] = 1  # set arbitrary BigM (ok, because we fix corresponding Y=True)

        del child_model.GDPopt_utils.disjunction_to_unfixed_disjuncts[child_disjunction_to_branch]
        for child_disjunct in child_unfixed_disjuncts:
            child_model.GDPopt_utils.disjunct_to_nonlinear_constraints.pop(child_disjunct, None)

        newly_created_nodes += 1
        child_node_data = node_data._replace(
            is_screened=False,
            is_evaluated=False,
            num_unbranched_disjunctions=node_data.num_unbranched_disjunctions - 1,
            node_count=node_count + newly_created_nodes,
            unbranched_disjunction_indices=node_data.unbranched_disjunction_indices[1:],
            obj_ub=float('inf'),
        )
        heappush(solve_data.bb_queue, (child_node_data, child_model))

    solve_data.created_nodes += newly_created_nodes

    config.logger.info("Added %s new nodes with %s relaxed disjunctions to the heap. Size now %s." % (
        num_unfixed_disjuncts, node_data.num_unbranched_disjunctions - 1, len(solve_data.bb_queue)))


def _prescreen_node(node_data, node_model, solve_data):
    config = solve_data.config
    # Check node for satisfiability if sat-solver is enabled
    if config.check_sat and satisfiable(node_model, config.logger) is False:
        if node_data.node_count == 0:
            config.logger.info("Root node is not satisfiable. Problem is infeasible.")
        else:
            config.logger.info("SAT solver pruned node %s" % node_data.node_count)
        new_lb = new_ub = float('inf')
    else:
        # Solve model subproblem
        if config.solve_local_rnGDP:
            solve_data.config.logger.debug(
                "Screening node %s with LB %.10g and %s inactive disjunctions." % (
                    node_data.node_count, node_data.obj_lb, node_data.num_unbranched_disjunctions
                ))
            new_lb, new_ub = _solve_local_rnGDP_subproblem(node_model, solve_data)
        else:
            new_lb, new_ub = float('-inf'), float('inf')
        new_lb = max(node_data.obj_lb, new_lb)

    new_node_data = node_data._replace(obj_lb=new_lb, obj_ub=new_ub, is_screened=True)
    return new_node_data


def _evaluate_node(node_data, node_model, solve_data):
    config = solve_data.config
    # Solve model subproblem
    solve_data.config.logger.info(
        "Exploring node %s with LB %.10g UB %.10g and %s inactive disjunctions." % (
            node_data.node_count, node_data.obj_lb, node_data.obj_ub, node_data.num_unbranched_disjunctions
        ))
    new_lb, new_ub = _solve_rnGDP_subproblem(node_model, solve_data)
    new_node_data = node_data._replace(obj_lb=new_lb, obj_ub=new_ub, is_evaluated=True)
    return new_node_data


def _solve_rnGDP_subproblem(model, solve_data):
    config = solve_data.config
    subproblem = TransformationFactory('gdp.bigm').create_using(model)
    obj_sense_correction = solve_data.objective_sense != minimize

    try:
        with SuppressInfeasibleWarning():
            try:
                fbbt(subproblem, integer_tol=config.integer_tolerance)
            except InfeasibleConstraintException:
                copy_var_list_values(  # copy variable values, even if errored
                    from_list=subproblem.GDPopt_utils.variable_list,
                    to_list=model.GDPopt_utils.variable_list,
                    config=config, ignore_integrality=True
                )
                return float('inf'), float('inf')
            minlp_args = dict(config.minlp_solver_args)
            if config.minlp_solver == 'gams':
                elapsed = get_main_elapsed_time(solve_data.timing)
                remaining = max(config.time_limit - elapsed, 1)
                minlp_args['add_options'] = minlp_args.get('add_options', [])
                minlp_args['add_options'].append('option reslim=%s;' % remaining)
            result = SolverFactory(config.minlp_solver).solve(subproblem, **minlp_args)
    except RuntimeError as e:
        config.logger.warning(
            "Solver encountered RuntimeError. Treating as infeasible. "
            "Msg: %s\n%s" % (str(e), traceback.format_exc()))
        copy_var_list_values(  # copy variable values, even if errored
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('inf'), float('inf')

    term_cond = result.solver.termination_condition
    if term_cond == tc.optimal:
        assert result.solver.status is SolverStatus.ok
        lb = result.problem.lower_bound if not obj_sense_correction else -result.problem.upper_bound
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config,
        )
        return lb, ub
    elif term_cond == tc.locallyOptimal or term_cond == tc.feasible:
        assert result.solver.status is SolverStatus.ok
        lb = result.problem.lower_bound if not obj_sense_correction else -result.problem.upper_bound
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        # TODO handle LB absent
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config,
        )
        return lb, ub
    elif term_cond == tc.unbounded:
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('-inf'), float('-inf')
    elif term_cond == tc.infeasible:
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('inf'), float('inf')
    else:
        config.logger.warning("Unknown termination condition of %s. Treating as infeasible." % term_cond)
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('inf'), float('inf')


def _solve_local_rnGDP_subproblem(model, solve_data):
    # TODO for now, return (LB, UB) = (-inf, inf) (for minimize)
    config = solve_data.config
    subproblem = TransformationFactory('gdp.bigm').create_using(model)
    obj_sense_correction = solve_data.objective_sense != minimize

    try:
        with SuppressInfeasibleWarning():
            result = SolverFactory(config.local_minlp_solver).solve(subproblem, **config.local_minlp_solver_args)
    except RuntimeError as e:
        config.logger.warning(
            "Solver encountered RuntimeError. Treating as infeasible. "
            "Msg: %s\n%s" % (str(e), traceback.format_exc()))
        copy_var_list_values(  # copy variable values, even if errored
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('-inf'), float('inf')

    term_cond = result.solver.termination_condition
    if term_cond == tc.optimal:
        assert result.solver.status is SolverStatus.ok
        lb = result.problem.lower_bound if not obj_sense_correction else -result.problem.upper_bound
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config,
        )
        return float('-inf'), ub
    elif term_cond == tc.locallyOptimal or term_cond == tc.feasible:
        assert result.solver.status is SolverStatus.ok
        lb = result.problem.lower_bound if not obj_sense_correction else -result.problem.upper_bound
        ub = result.problem.upper_bound if not obj_sense_correction else -result.problem.lower_bound
        # TODO handle LB absent
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config,
        )
        return float('-inf'), ub
    elif term_cond == tc.unbounded:
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('-inf'), float('-inf')
    elif term_cond == tc.infeasible:
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('-inf'), float('inf')
    else:
        config.logger.warning("Unknown termination condition of %s. Treating as infeasible." % term_cond)
        copy_var_list_values(
            from_list=subproblem.GDPopt_utils.variable_list,
            to_list=model.GDPopt_utils.variable_list,
            config=config, ignore_integrality=True
        )
        return float('-inf'), float('inf')
