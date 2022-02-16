"""Module for GDPopt data classes."""
from pyomo.common.collections import Bunch
from pyomo.opt import SolverResults, ProblemSense
from pyomo.util.model_size import build_model_size_report
from pyomo.core.base import Objective


class GDPoptSolveData(object):
    """Data container to hold solve-instance data.

    Attributes:
        - results (SolverResults): Pyomo results objective
        - timing (Bunch): dictionary of time elapsed for solver functions

    """
    pass

class AlgorithmProgress(object):
    """Data container to track progress of algorithm

    Attributes:
        - dual_bound
        - primal_bound
        - iteration_log
    """
    def __init__(self, original_model, config, solver_version):
        self.LB = -float('inf')
        self.UB = float('inf')
        self.iteration_log = {}
        self.timing = Bunch()
        # TODO: rename these when you understand what they are
        self.master_iteration = 0
        self.mip_iteration = 0
        self.nlp_iteration = 0
        self.pyomo_results = self.get_pyomo_results_object_with_problem_info(
            original_model, config, solver_version)
        self.obj_sense = self.pyomo_results.problem.sense
        self.incumbent = None
        self.primal_bound_improved = False

    def update_bounds(primal=None, dual=None):
        """
        Update bounds correctly depending on objective sense.

        primal: bound from solving subproblem with fixed master solution
        dual: bound from solving master problem (relaxation of original problem)
        """
        if self.obj_sense is minimize:
            if primal is not None:
                self.UB = primal
            if dual is not None:
                self.LB = dual
        else:
            if primal is not None:
                self.LB = primal
            if dual is not None:
                self.UB = dual

    def get_pyomo_results_object_with_problem_info(self, original_model, config,
                                                   solver_version):
        """
        Initialize a results object with results.problem information
        """
        results = SolverResults()

        results.solver.name = 'GDPopt %s - %s' % (solver_version,
                                                  config.strategy)
    
        prob = results.problem
        prob.name = original_model.name
        prob.number_of_nonzeros = None  # TODO

        num_of = build_model_size_report(original_model)

        # Get count of constraints and variables
        prob.number_of_constraints = num_of.activated.constraints
        prob.number_of_disjunctions = num_of.activated.disjunctions
        prob.number_of_variables = num_of.activated.variables
        prob.number_of_binary_variables = num_of.activated.binary_variables
        prob.number_of_continuous_variables = num_of.activated.\
                                              continuous_variables
        prob.number_of_integer_variables = num_of.activated.integer_variables

        config.logger.info(
            "Original model has %s constraints (%s nonlinear) "
            "and %s disjunctions, "
            "with %s variables, of which %s are binary, %s are integer, "
            "and %s are continuous." %
            (num_of.activated.constraints,
             num_of.activated.nonlinear_constraints,
             num_of.activated.disjunctions,
             num_of.activated.variables,
             num_of.activated.binary_variables,
             num_of.activated.integer_variables,
             num_of.activated.continuous_variables))

        # Handle missing or multiple objectives, and get sense
        active_objectives = list(original_model.component_data_objects(
            ctype=Objective, active=True, descend_into=True))
        number_of_objectives = len(active_objectives)
        if number_of_objectives == 0:
            logger.warning(
                'Model has no active objectives. Adding dummy objective.')
            main_obj = gdpopt_block.dummy_objective = Objective(expr=1)
        elif number_of_objectives > 1:
            raise ValueError('Model has multiple active objectives.')
        else:
            main_obj = active_objectives[0]
        results.problem.sense = ProblemSense.minimize if main_obj.sense == 1 \
                                else ProblemSense.maximize

        return results

    def get_final_pyomo_results_object(self):
        """
        Fill in the results.solver information onto the results object
        """
        results = self.pyomo_results
        # Finalize results object
        results.problem.lower_bound = self.LB
        results.problem.upper_bound = self.UB
        results.solver.iterations = self.master_iteration
        results.solver.timing = self.timing
        results.solver.user_time = self.timing.total
        results.solver.wallclock_time = self.timing.total

        return results

# ESJ TODO: I don't think we need the var values here or the disjunct_values
# because they are on the model object. It's not clear we need an object...
class MasterProblemResult(object):
    """Data class for master problem results data.

    Key attributes:
        - feasible: True/False if feasible solution obtained
        - var_values: list of variable values
        - pyomo_results: results object from solve() statement
        - disjunct_values: list of disjunct values

    """
    pass


class SubproblemResult(object):
    """Data class for subproblem results data.

    Key attributes:
        - feasible: True/False if feasible solution obtained
        - var_values: list of variable values
        - dual_values: list of constraint dual values
        - pyomo_results: results object from solve() statement

    """
