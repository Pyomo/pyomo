"""Main driver module for GDPbb solver.

19.5.7 changes:
- added support for time limits
- rewrote algorithm to solve linear relaxation at root node and improve stability

"""

import heapq
import logging
import traceback

from pyomo.common import deprecated
from pyomo.common.collections import ComponentSet, Container
from pyomo.common.config import (ConfigBlock, ConfigValue, PositiveInt)
from pyomo.contrib.gdpopt.util import create_utility_block, time_code, a_logger, restore_logger_level, \
    setup_results_object, get_main_elapsed_time, process_objective
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import (
    Objective, TransformationFactory,
    minimize, value, Constraint, Suffix)
from pyomo.gdp import Disjunction
from pyomo.opt import SolverFactory, SolverStatus, SolverResults
from pyomo.opt import TerminationCondition as tc

__version__ = (19, 5, 7)  # Date-based versioning


class GDPbbSolveData(object):
    pass


@SolverFactory.register('gdpbb',
                        doc='Branch and Bound based GDP Solver')
class GDPbbSolver(object):
    """
    A branch and bound-based solver for Generalized Disjunctive Programming (GDP) problems

    The GDPbb solver solves subproblems relaxing certain disjunctions, and
    builds up a tree of potential active disjunctions. By exploring promising
    branches, it eventually results in an optimal configuration of disjunctions.

    Keyword arguments below are specified for the ``solve`` function.

    """
    CONFIG = ConfigBlock("gdpbb")
    CONFIG.declare("solver", ConfigValue(
        default="baron",
        description="Subproblem solver to use, defaults to baron"
    ))
    CONFIG.declare("solver_args", ConfigBlock(
        implicit=True,
        description="Block of keyword arguments to pass to the solver."
    ))
    CONFIG.declare("tee", ConfigValue(
        default=False,
        domain=bool,
        description="Flag to stream solver output to console."
    ))
    CONFIG.declare("check_sat", ConfigValue(
        default=False,
        domain=bool,
        description="When True, GDPBB will check satisfiability via the pyomo.contrib.satsolver interface at each node"
    ))
    CONFIG.declare("logger", ConfigValue(
        default='pyomo.contrib.gdpbb',
        description="The logger object or name to use for reporting.",
        domain=a_logger
    ))
    CONFIG.declare("time_limit", ConfigValue(
        default=600,
        domain=PositiveInt,
        description="Time limit (seconds, default=600)",
        doc="Seconds allowed until terminated. Note that the time limit can"
            "currently only be enforced between subsolver invocations. You may"
            "need to set subsolver time limits as well."
    ))

    @deprecated("GDPbb has been merged into GDPopt. "
                "You can use the algorithm using GDPopt with strategy='LBB'.",
                logger="pyomo.solvers",
                version='5.6.9')
    def __init__(self, *args, **kwargs):
        super(GDPbbSolver, self).__init__(*args, **kwargs)

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def license_is_valid(self):
        return True

    def version(self):
        return __version__

    def solve(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)
        return SolverFactory('gdpopt').solve(
            model, strategy='LBB',
            minlp_solver=config.solver, minlp_solver_args=config.solver_args,
            tee=config.tee, check_sat=config.check_sat, logger=config.logger,
            time_limit=config.time_limit
        )

        # Validate model to be used with gdpbb
        self.validate_model(model)
        # Set solver as an MINLP
        solve_data = GDPbbSolveData()
        solve_data.timing = Container()
        solve_data.original_model = model
        solve_data.results = SolverResults()

        old_logger_level = config.logger.getEffectiveLevel()
        with time_code(solve_data.timing, 'total', is_main_timer=True), \
                restore_logger_level(config.logger), \
                create_utility_block(model, 'GDPbb_utils', solve_data):
            if config.tee and old_logger_level > logging.INFO:
                # If the logger does not already include INFO, include it.
                config.logger.setLevel(logging.INFO)
            config.logger.info(
                "Starting GDPbb version %s using %s as subsolver"
                % (".".join(map(str, self.version())), config.solver)
            )

            # Setup results
            solve_data.results.solver.name = 'GDPbb - %s' % (str(config.solver))
            setup_results_object(solve_data, config)

            # clone original model for root node of branch and bound
            root = solve_data.working_model = solve_data.original_model.clone()

            # get objective sense
            process_objective(solve_data, config)
            objectives = solve_data.original_model.component_data_objects(Objective, active=True)
            obj = next(objectives, None)
            solve_data.results.problem.sense = obj.sense

            # set up lists to keep track of which disjunctions have been covered.

            # this list keeps track of the relaxed disjunctions
            root.GDPbb_utils.unenforced_disjunctions = list(
                disjunction for disjunction in root.GDPbb_utils.disjunction_list if disjunction.active
            )

            root.GDPbb_utils.deactivated_constraints = ComponentSet([
                constr for disjunction in root.GDPbb_utils.unenforced_disjunctions
                for disjunct in disjunction.disjuncts
                for constr in disjunct.component_data_objects(ctype=Constraint, active=True)
                if constr.body.polynomial_degree() not in (1, 0)
            ])
            # Deactivate nonlinear constraints in unenforced disjunctions
            for constr in root.GDPbb_utils.deactivated_constraints:
                constr.deactivate()

            # Add the BigM suffix if it does not already exist. Used later during nonlinear constraint activation.
            if not hasattr(root, 'BigM'):
                root.BigM = Suffix()

            # Pre-screen that none of the disjunctions are already predetermined due to the disjuncts being fixed
            # to True/False values.
            # TODO this should also be done within the loop, but we aren't handling it right now.
            # Should affect efficiency, but not correctness.
            root.GDPbb_utils.disjuncts_fixed_True = ComponentSet()
            # Only find top-level (non-nested) disjunctions
            for disjunction in root.component_data_objects(Disjunction, active=True):
                fixed_true_disjuncts = [disjunct for disjunct in disjunction.disjuncts
                                        if disjunct.indicator_var.fixed
                                        and disjunct.indicator_var.value == 1]
                fixed_false_disjuncts = [disjunct for disjunct in disjunction.disjuncts
                                         if disjunct.indicator_var.fixed
                                         and disjunct.indicator_var.value == 0]
                for disjunct in fixed_false_disjuncts:
                    disjunct.deactivate()
                if len(fixed_false_disjuncts) == len(disjunction.disjuncts) - 1:
                    # all but one disjunct in the disjunction is fixed to False. Remaining one must be true.
                    if not fixed_true_disjuncts:
                        fixed_true_disjuncts = [disjunct for disjunct in disjunction.disjuncts
                                                if disjunct not in fixed_false_disjuncts]
                # Reactivate the fixed-true disjuncts
                for disjunct in fixed_true_disjuncts:
                    newly_activated = ComponentSet()
                    for constr in disjunct.component_data_objects(Constraint):
                        if constr in root.GDPbb_utils.deactivated_constraints:
                            newly_activated.add(constr)
                            constr.activate()
                            # Set the big M value for the constraint
                            root.BigM[constr] = 1
                            # Note: we use a default big M value of 1
                            # because all non-selected disjuncts should be deactivated.
                            # Therefore, none of the big M transformed nonlinear constraints will need to be relaxed.
                            # The default M value should therefore be irrelevant.
                    root.GDPbb_utils.deactivated_constraints -= newly_activated
                    root.GDPbb_utils.disjuncts_fixed_True.add(disjunct)

                if fixed_true_disjuncts:
                    assert disjunction.xor, "GDPbb only handles disjunctions in which one term can be selected. " \
                        "%s violates this assumption." % (disjunction.name, )
                    root.GDPbb_utils.unenforced_disjunctions.remove(disjunction)

            # Check satisfiability
            if config.check_sat and satisfiable(root, config.logger) is False:
                # Problem is not satisfiable. Problem is infeasible.
                obj_value = obj_sign * float('inf')
            else:
                # solve the root node
                config.logger.info("Solving the root node.")
                obj_value, result, var_values = self.subproblem_solve(root, config)

            if obj_sign * obj_value == float('inf'):
                config.logger.info("Model was found to be infeasible at the root node. Elapsed %.2f seconds."
                                   % get_main_elapsed_time(solve_data.timing))
                if solve_data.results.problem.sense == minimize:
                    solve_data.results.problem.lower_bound = float('inf')
                    solve_data.results.problem.upper_bound = None
                else:
                    solve_data.results.problem.lower_bound = None
                    solve_data.results.problem.upper_bound = float('-inf')
                solve_data.results.solver.timing = solve_data.timing
                solve_data.results.solver.iterations = 0
                solve_data.results.solver.termination_condition = tc.infeasible
                return solve_data.results

            # initialize minheap for Branch and Bound algorithm
            # Heap structure: (ordering tuple, model)
            # Ordering tuple: (objective value, disjunctions_left, -total_nodes_counter)
            #  - select solutions with lower objective value,
            #    then fewer disjunctions left to explore (depth first),
            #    then more recently encountered (tiebreaker)
            heap = []
            total_nodes_counter = 0
            disjunctions_left = len(root.GDPbb_utils.unenforced_disjunctions)
            heapq.heappush(
                heap, (
                    (obj_sign * obj_value, disjunctions_left, -total_nodes_counter),
                    root, result, var_values))

            # loop to branch through the tree
            while len(heap) > 0:
                # pop best model off of heap
                sort_tuple, incumbent_model, incumbent_results, incumbent_var_values = heapq.heappop(heap)
                incumbent_obj_value, disjunctions_left, _ = sort_tuple

                config.logger.info("Exploring node with LB %.10g and %s inactive disjunctions." % (
                    incumbent_obj_value, disjunctions_left
                ))

                # if all the originally active disjunctions are active, solve and
                # return solution
                if disjunctions_left == 0:
                    config.logger.info("Model solved.")
                    # Model is solved. Copy over solution values.
                    original_model = solve_data.original_model
                    for orig_var, val in zip(original_model.GDPbb_utils.variable_list, incumbent_var_values):
                        orig_var.value = val

                    solve_data.results.problem.lower_bound = incumbent_results.problem.lower_bound
                    solve_data.results.problem.upper_bound = incumbent_results.problem.upper_bound
                    solve_data.results.solver.timing = solve_data.timing
                    solve_data.results.solver.iterations = total_nodes_counter
                    solve_data.results.solver.termination_condition = incumbent_results.solver.termination_condition
                    return solve_data.results

                # Pick the next disjunction to branch on
                next_disjunction = incumbent_model.GDPbb_utils.unenforced_disjunctions[0]
                config.logger.info("Branching on disjunction %s" % next_disjunction.name)
                assert next_disjunction.xor, "GDPbb only handles disjunctions in which one term can be selected. " \
                    "%s violates this assumption." % (next_disjunction.name, )

                new_nodes_counter = 0

                for i, disjunct in enumerate(next_disjunction.disjuncts):
                    # Create one branch for each of the disjuncts on the disjunction

                    if any(disj.indicator_var.fixed and disj.indicator_var.value == 1
                           for disj in next_disjunction.disjuncts if disj is not disjunct):
                        # If any other disjunct is fixed to 1 and an xor relationship applies,
                        # then this disjunct cannot be activated.
                        continue

                    # Check time limit
                    if get_main_elapsed_time(solve_data.timing) >= config.time_limit:
                        if solve_data.results.problem.sense == minimize:
                            solve_data.results.problem.lower_bound = incumbent_obj_value
                            solve_data.results.problem.upper_bound = float('inf')
                        else:
                            solve_data.results.problem.lower_bound = float('-inf')
                            solve_data.results.problem.upper_bound = incumbent_obj_value
                        config.logger.info(
                            'GDPopt unable to converge bounds '
                            'before time limit of {} seconds. '
                            'Elapsed: {} seconds'
                            .format(config.time_limit, get_main_elapsed_time(solve_data.timing)))
                        config.logger.info(
                            'Final bound values: LB: {}  UB: {}'.
                            format(solve_data.results.problem.lower_bound, solve_data.results.problem.upper_bound))
                        solve_data.results.solver.timing = solve_data.timing
                        solve_data.results.solver.iterations = total_nodes_counter
                        solve_data.results.solver.termination_condition = tc.maxTimeLimit
                        return solve_data.results

                    # Branch on the disjunct
                    child = incumbent_model.clone()
                    # TODO I am leaving the old branching system in place, but there should be
                    # something better, ideally that deals with nested disjunctions as well.
                    disjunction_to_branch = child.GDPbb_utils.unenforced_disjunctions.pop(0)
                    child_disjunct = disjunction_to_branch.disjuncts[i]
                    child_disjunct.indicator_var.fix(1)
                    # Deactivate (and fix to 0) other disjuncts on the disjunction
                    for disj in disjunction_to_branch.disjuncts:
                        if disj is not child_disjunct:
                            disj.deactivate()
                    # Activate nonlinear constraints on the newly fixed child disjunct
                    newly_activated = ComponentSet()
                    for constr in child_disjunct.component_data_objects(Constraint):
                        if constr in child.GDPbb_utils.deactivated_constraints:
                            newly_activated.add(constr)
                            constr.activate()
                            # Set the big M value for the constraint
                            child.BigM[constr] = 1
                            # Note: we use a default big M value of 1
                            # because all non-selected disjuncts should be deactivated.
                            # Therefore, none of the big M transformed nonlinear constraints will need to be relaxed.
                            # The default M value should therefore be irrelevant.
                    child.GDPbb_utils.deactivated_constraints -= newly_activated
                    child.GDPbb_utils.disjuncts_fixed_True.add(child_disjunct)

                    if disjunct in incumbent_model.GDPbb_utils.disjuncts_fixed_True:
                        # If the disjunct was already branched to True from a parent disjunct branching, just pass
                        # through the incumbent value without resolving. The solution should be the same as the parent.
                        total_nodes_counter += 1
                        ordering_tuple = (obj_sign * incumbent_obj_value, disjunctions_left - 1, -total_nodes_counter)
                        heapq.heappush(heap, (ordering_tuple, child, result, incumbent_var_values))
                        new_nodes_counter += 1
                        continue

                    if config.check_sat and satisfiable(child, config.logger) is False:
                        # Problem is not satisfiable. Skip this disjunct.
                        continue

                    obj_value, result, var_values = self.subproblem_solve(child, config)
                    total_nodes_counter += 1
                    ordering_tuple = (obj_sign * obj_value, disjunctions_left - 1, -total_nodes_counter)
                    heapq.heappush(heap, (ordering_tuple, child, result, var_values))
                    new_nodes_counter += 1

                config.logger.info("Added %s new nodes with %s relaxed disjunctions to the heap. Size now %s." % (
                    new_nodes_counter, disjunctions_left - 1, len(heap)))

    @staticmethod
    def validate_model(model):
        # Validates that model has only exclusive disjunctions
        for d in model.component_data_objects(
                ctype=Disjunction, active=True):
            if not d.xor:
                raise ValueError('GDPbb solver unable to handle '
                                 'non-exclusive disjunctions')
        objectives = model.component_data_objects(Objective, active=True)
        obj = next(objectives, None)
        if next(objectives, None) is not None:
            raise RuntimeError(
                "GDPbb solver is unable to handle model with multiple active objectives.")
        if obj is None:
            raise RuntimeError(
                "GDPbb solver is unable to handle model with no active objective.")

    @staticmethod
    def subproblem_solve(gdp, config):
        subproblem = gdp.clone()
        TransformationFactory('gdp.bigm').apply_to(subproblem)
        main_obj = next(subproblem.component_data_objects(Objective, active=True))
        obj_sign = 1 if main_obj.sense == minimize else -1

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

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
