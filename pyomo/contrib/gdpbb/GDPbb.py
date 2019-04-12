import heapq
import logging

from pyutilib.misc import Container

from pyomo.common.config import (ConfigBlock, ConfigValue)
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.util import create_utility_block, time_code, a_logger, restore_logger_level, \
    setup_results_object
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core.base import (
    Objective, TransformationFactory,
    minimize, value)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory, SolverStatus, SolverResults
from pyomo.opt import TerminationCondition as tc

__version__ = (19, 2, 19)  # Date-based versioning


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

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

    def version(self):
        return __version__

    def solve(self, model, **kwds):
        config = self.CONFIG(kwds.pop('options', {}))
        config.set_value(kwds)

        # Validate model to be used with gdpbb
        self.validate_model(model)
        # Set solver as an MINLP
        solver = SolverFactory(config.solver)
        solve_data = GDPbbSolveData()
        solve_data.timing = Container()
        solve_data.original_model = model
        solve_data.results = SolverResults()

        old_logger_level = config.logger.getEffectiveLevel()
        with time_code(solve_data.timing, 'total'), \
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
            # Initialize list containing indicator vars for reupdating model after solving
            indicator_list_name = unique_component_name(model, "_indicator_list")
            indicator_vars = []
            for disjunction in model.component_data_objects(
                    ctype=Disjunction, active=True):
                for disjunct in disjunction.disjuncts:
                    indicator_vars.append(disjunct.indicator_var)
            setattr(model, indicator_list_name, indicator_vars)

            # get objective sense
            objectives = model.component_data_objects(Objective, active=True)
            obj = next(objectives, None)
            obj_sign = 1 if obj.sense == minimize else -1
            solve_data.results.problem.sense = obj.sense
            # clone original model for root node of branch and bound
            root = model.clone()

            # set up lists to keep track of which disjunctions have been covered.

            # this list keeps track of the original disjunctions that were active and are soon to be inactive
            root.GDPbb_utils.unenforced_disjunctions = list(
                disjunction for disjunction in root.GDPbb_utils.disjunction_list if disjunction.active
            )

            # this list keeps track of the disjunctions that have been activated by the branch and bound
            root.GDPbb_utils.curr_active_disjunctions = []

            # deactivate all disjunctions in the model
            # self.indicate(root)
            for djn in root.GDPbb_utils.unenforced_disjunctions:
                djn.deactivate()
            # Deactivate all disjuncts in model. To be reactivated when disjunction
            # is reactivated.
            for disj in root.component_data_objects(Disjunct, active=True):
                disj._deactivate_without_fixing_indicator()

            # Satisfiability check would go here

            # solve the root node
            config.logger.info("Solving the root node.")
            obj_value, result, _ = self.subproblem_solve(root, solver, config)

            # initialize minheap for Branch and Bound algorithm
            # Heap structure: (ordering tuple, model)
            # Ordering tuple: (objective value, disjunctions_left, -counter)
            #  - select solutions with lower objective value,
            #    then fewer disjunctions left to explore (depth first),
            #    then more recently encountered (tiebreaker)
            heap = []
            counter = 0
            disjunctions_left = len(root.GDPbb_utils.unenforced_disjunctions)
            heapq.heappush(heap,
                           ((obj_sign * obj_value, disjunctions_left, -counter), root,
                            result, root.GDPbb_utils.variable_list))
            # loop to branch through the tree
            while len(heap) > 0:
                # pop best model off of heap
                sort_tup, mdl, mdl_results, vars = heapq.heappop(heap)
                old_obj_val, disjunctions_left, _ = sort_tup
                config.logger.info("Exploring node with LB %.10g and %s inactive disjunctions." % (
                    old_obj_val, disjunctions_left
                ))

                # if all the originally active disjunctions are active, solve and
                # return solution
                if disjunctions_left == 0:
                    config.logger.info("Model solved.")
                    # Model is solved. Copy over solution values.
                    for orig_var, soln_var in zip(model.GDPbb_utils.variable_list, vars):
                        orig_var.value = soln_var.value

                    solve_data.results.problem.lower_bound = mdl_results.problem.lower_bound
                    solve_data.results.problem.upper_bound = mdl_results.problem.upper_bound
                    solve_data.results.solver.timing = solve_data.timing
                    solve_data.results.solver.termination_condition = mdl_results.solver.termination_condition
                    return solve_data.results

                next_disjunction = mdl.GDPbb_utils.unenforced_disjunctions.pop(0)
                config.logger.info("Activating disjunction %s" % next_disjunction.name)
                next_disjunction.activate()
                mdl.GDPbb_utils.curr_active_disjunctions.append(next_disjunction)
                djn_left = len(mdl.GDPbb_utils.unenforced_disjunctions)
                for disj in next_disjunction.disjuncts:
                    disj._activate_without_unfixing_indicator()
                    if not disj.indicator_var.fixed:
                        disj.indicator_var = 0  # initially set all indicator vars to zero
                added_disj_counter = 0
                for disj in next_disjunction.disjuncts:
                    if not disj.indicator_var.fixed:
                        disj.indicator_var = 1
                    mnew = mdl.clone()
                    if not disj.indicator_var.fixed:
                        disj.indicator_var = 0

                    # Check feasibility
                    if config.check_sat and satisfiable(mnew, config.logger) is False:
                        # problem is not satisfiable. Skip this disjunct.
                        continue

                    obj_value, result, vars = self.subproblem_solve(mnew, solver, config)
                    counter += 1
                    ordering_tuple = (obj_sign * obj_value, djn_left, -counter)
                    heapq.heappush(heap, (ordering_tuple, mnew, result, vars))
                    added_disj_counter = added_disj_counter + 1
                config.logger.info("Added %s new nodes with %s relaxed disjunctions to the heap. Size now %s." % (
                    added_disj_counter, djn_left, len(heap)))

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
    def subproblem_solve(gdp, solver, config):
        subproblem = gdp.clone()
        TransformationFactory('gdp.fix_disjuncts').apply_to(subproblem)

        result = solver.solve(subproblem, **config.solver_args)
        main_obj = next(subproblem.component_data_objects(Objective, active=True))
        obj_sign = 1 if main_obj.sense == minimize else -1
        if (result.solver.status is SolverStatus.ok and
                result.solver.termination_condition is tc.optimal):
            return value(main_obj.expr), result, subproblem.GDPbb_utils.variable_list
        elif result.solver.termination_condition is tc.unbounded:
            return obj_sign * float('-inf'), result, subproblem.GDPbb_utils.variable_list
        else:
            return obj_sign * float('inf'), result, subproblem.GDPbb_utils.variable_list

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass
