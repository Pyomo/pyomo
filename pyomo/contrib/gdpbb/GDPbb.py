import heapq

from pyutilib.misc import Container

from pyomo.common.config import (ConfigBlock, ConfigValue)
from pyomo.common.modeling import unique_component_name
from pyomo.contrib.gdpopt.util import create_utility_block, time_code
from pyomo.core.base import (
    Objective, TransformationFactory,
    minimize, value)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory, SolverStatus, SolverResults
from pyomo.opt import TerminationCondition as tc


class GDPbbSolveData(object):
    pass


@SolverFactory.register('gdpbb',
                        doc='Branch and Bound based GDP Solver')
class GDPbbSolver(object):
    """A branch and bound-based GDP solver."""
    CONFIG = ConfigBlock("gdpbb")
    CONFIG.declare("solver", ConfigValue(
        default="baron",
        description="Subproblem solver to use, defaults to baron"
    ))
    CONFIG.declare("solver_args", ConfigValue(
        default={},
        description="Dictionary of keyword arguments to pass to the solver."
    ))
    CONFIG.declare("tee", ConfigValue(
        default=False,
        domain=bool,
        description="Flag to stream solver output to console."
    ))

    def available(self, exception_flag=True):
        """Check if solver is available.

        TODO: For now, it is always available. However, sub-solvers may not
        always be available, and so this should reflect that possibility.

        """
        return True

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

        with create_utility_block(model, 'GDPbb_utils', solve_data),\
                time_code(solve_data.timing, 'total'):
            # Initialize ist containing indicator vars for reupdating model after solving
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
            obj_value, result,_ = self.subproblem_solve(root, solver, config)

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
            result , root.GDPbb_utils.variable_list))
            # loop to branch through the tree
            while len(heap) > 0:
                # pop best model off of heap
                sort_tup, mdl, _, vars = heapq.heappop(heap)
                _, disjunctions_left, _ = sort_tup

                # if all the originally active disjunctions are active, solve and
                # return solution
                if disjunctions_left == 0:
                    # Model is solved. Copy over solution values.
                    for orig_var, soln_var in zip(model.GDPbb_utils.variable_list, vars):
                        orig_var.value = soln_var.value
                    return solve_data.results

                next_disjunction = mdl.GDPbb_utils.unenforced_disjunctions.pop(0)
                next_disjunction.activate()
                mdl.GDPbb_utils.curr_active_disjunctions.append(next_disjunction)
                for disj in next_disjunction.disjuncts:
                    disj._activate_without_unfixing_indicator()
                    if not disj.indicator_var.fixed:
                        disj.indicator_var = 0  # initially set all indicator vars to zero
                for disj in next_disjunction.disjuncts:
                    if not disj.indicator_var.fixed:
                        disj.indicator_var = 1
                    mnew = mdl.clone()
                    if not disj.indicator_var.fixed:
                        disj.indicator_var = 0
                    obj_value, result, vars = self.subproblem_solve(mnew, solver, config)
                    counter += 1
                    djn_left = len(mdl.GDPbb_utils.unenforced_disjunctions)
                    ordering_tuple = (obj_sign * obj_value, djn_left, -counter)
                    heapq.heappush(heap, (ordering_tuple, mnew, result, vars))

    def validate_model(self, model):
        # Validates that model has only exclusive disjunctions
        for d in model.component_data_objects(
                ctype=Disjunction, active=True):
            if (not d.xor):
                raise ValueError('GDPlbb unable to handle '
                                 'non-exclusive disjunctions')
        objectives = model.component_data_objects(Objective, active=True)
        obj = next(objectives, None)
        if next(objectives, None) is not None:
            raise RuntimeError(
                "GDP LBB solver is unable to handle model with multiple active objectives.")
        if obj is None:
            raise RuntimeError(
                "GDP LBB solver is unable to handle model with no active objective.")

    def subproblem_solve(self, gdp, solver, config):
        subproblem = gdp.clone()
        TransformationFactory('gdp.fix_disjuncts').apply_to(subproblem)
        for disjunct in subproblem.component_data_objects(
                ctype=Disjunct, active=True):
            disjunct.deactivate()  # TODO this is HACK

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

    def indicate(self, model):
        for disjunction in model.component_data_objects(
                ctype=Disjunction,active = True):
            for disjunct in disjunction.disjuncts:
                print(disjunction.name, disjunct.name, disjunct.indicator_var.value)
