import heapq

from pyomo.common.config import (ConfigBlock, ConfigValue)
from pyomo.common.modeling import unique_component_name
from pyomo.core.base import (
    Objective, TransformationFactory,
    minimize, value)
from pyomo.gdp import Disjunct, Disjunction
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition as tc


@SolverFactory.register('gdpbb',
                        doc='Branch and Bound based GDP Solver')
class GDPbbSolver(object):
    """A branch and bound-based GDP solver."""
    CONFIG = ConfigBlock("gdpbb")
    CONFIG.declare("solver", ConfigValue(
        default="baron",
        description="Solver to use, defaults to baron"
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
        print(obj_sign)

        # clone original model for root node of branch and bound
        root = model.clone()

        # set up lists to keep track of which disjunctions have been covered.

        # this list keeps track of the original disjunctions that were active and are soon to be inactive
        init_active_disjunctions_name = unique_component_name(root, "_init_active_disjunctions")
        init_active_disjunctions = list(root.component_data_objects(
            ctype=Disjunction, active=True))
        setattr(root, init_active_disjunctions_name, init_active_disjunctions)

        # this list keeps track of the disjunctions that have been activated by the branch and bound
        curr_active_disjunctions_name = unique_component_name(root, "_curr_active_disjunctions")
        curr_active_disjunctions = []
        setattr(root, curr_active_disjunctions_name, curr_active_disjunctions)

        # deactivate all disjunctions in the model
        # self.indicate(root)
        for djn in getattr(root, init_active_disjunctions_name):
            djn.deactivate()
        # Deactivate all disjuncts in model. To be reactivated with disjunction
        # is reactivated.
        for disj in root.component_data_objects(Disjunct, active=True):
            disj.deactivate()
            disj.indicator_var
        for djn in getattr(root, init_active_disjunctions_name):
            djn.disjuncts[0].indicator_var = 1

        self.indicate(root)
        # Satisfiability check would go here

        # solve the root node
        obj_value = self.minlp_solve(root, solver, config)

        # initialize minheap for Branch and Bound algorithm
        heap = []
        heapq.heappush(heap, (obj_sign * obj_value, root))
        print([i[0] for i in heap])
        # loop to branch through the tree
        n = 0
        while len(heap) > 0:
            n = n + 1
            # pop best model off of heap
            mdlpack = heapq.heappop(heap)
            # print [i[0] for i in heap]
            mdl = mdlpack[1]

            print(mdlpack[0])
            # if all the originally active disjunctions are active, solve and
            # return solution
            if (len(getattr(mdl, init_active_disjunctions_name)) == 0):
                orig_var_list = getattr(model, indicator_list_name)
                best_soln_var_list = getattr(mdl, indicator_list_name)
                for orig_var, new_var in zip(orig_var_list, best_soln_var_list):
                    if not orig_var.is_fixed():
                        orig_var.value = new_var.value
                TransformationFactory('gdp.fix_disjuncts').apply_to(model)

                return solver.solve(model, **config.solver_args)

            disjunction = getattr(mdl, init_active_disjunctions_name).pop(0)
            for disj in list(disjunction.disjuncts):
                disj.activate()
            disjunction.activate()
            getattr(mdl, curr_active_disjunctions_name).append(disjunction)
            for disj in list(disjunction.disjuncts):
                disj.indicator_var = 0
            for disj in list(disjunction.disjuncts):
                disj.indicator_var = 1
                mnew = mdl.clone()
                disj.indicator_var = 0
                obj_value = self.minlp_solve(mnew, solver, config)
                print([value(d.indicator_var) for d in mnew.component_data_objects(Disjunct, active=True)])
                # self.indicate(mnew)
                djn_left = len(getattr(mdl, init_active_disjunctions_name))
                ordering_tuple = (obj_sign * obj_value, djn_left, -n)
                heapq.heappush(heap, (ordering_tuple, mnew))

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

    def minlp_solve(self, gdp, solver, config):
        minlp = gdp.clone()
        TransformationFactory('gdp.fix_disjuncts').apply_to(minlp)
        for disjunct in minlp.component_data_objects(
                ctype=Disjunct, active=True):
            disjunct.deactivate()  # TODO this is HACK

        result = solver.solve(minlp, **config.solver_args)
        if (result.solver.status is SolverStatus.ok and
                result.solver.termination_condition is tc.optimal):
            objectives = minlp.component_data_objects(Objective, active=True)
            obj = next(objectives, None)
            return value(obj.expr)
        else:
            objectives = minlp.component_data_objects(Objective, active=True)
            obj = next(objectives, None)
            obj_sign = 1 if obj.sense == minimize else -1
            return obj_sign * float('inf')

    def __enter__(self):
        return self

    def __exit__(self, t, v, traceback):
        pass

    def indicate(self, model):
        for disjunction in model.component_data_objects(
                ctype=Disjunction):
            for disjunct in disjunction.disjuncts:
                print(disjunction.name, disjunct.name, value(disjunct.indicator_var))
