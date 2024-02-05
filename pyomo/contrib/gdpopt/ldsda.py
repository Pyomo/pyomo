#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from collections import namedtuple
from heapq import heappush, heappop
import traceback

from pyomo.common.collections import ComponentMap
from pyomo.common.config import document_kwargs_from_configdict
from pyomo.common.errors import InfeasibleConstraintException
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.contrib.gdpopt.algorithm_base_class import _GDPoptAlgorithm
from pyomo.contrib.gdpopt.create_oa_subproblems import (
    add_util_block,
    add_disjunction_list,
    add_disjunct_list,
    add_algebraic_variable_list,
    add_boolean_variable_lists,
    add_transformed_boolean_variable_list,
)
from pyomo.contrib.gdpopt.config_options import (
    _add_nlp_solver_configs,
    _add_BB_configs,
    _add_ldsda_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import (
    copy_var_list_values,
    SuppressInfeasibleWarning,
    get_main_elapsed_time,
)
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import (
    minimize,
    Suffix,
    Constraint,
    TransformationFactory,
    BooleanVar,
    Var,
    Objective,
)
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.common.dependencies import attempt_import

it, it_available = attempt_import('itertools')

_linear_degrees = {1, 0}

# Data tuple for each node that also functions as the sort key.
# Therefore, ordering of the arguments below matters.
BBNodeData = namedtuple(
    'BBNodeData',
    [
        'obj_lb',  # lower bound on objective value, sign corrected to minimize
        'obj_ub',  # upper bound on objective value, sign corrected to minimize
        'is_screened',  # True if the node has been screened; False if not.
        'is_evaluated',  # True if node has been evaluated; False if not.
        'num_unbranched_disjunctions',  # number of unbranched disjunctions
        'node_count',  # cumulative node counter
        'unbranched_disjunction_indices',  # list of unbranched disjunction indices
    ],
)

ExternalVarInfo = namedtuple(
    'ExternalVarInfo',
    [
        'exactly_number',  # number of external variables for this type
        'Boolean_vars',  # list with names of the ordered Boolean variables to be reformulated
        'Disjuncts',  # list of disjuncts that are associated with the external variables
        # 'Boolean_vars_ordered_index',  # Indexes where the external reformulation is applied
        'LogicExpression',  # Logic expression that defines the external variables
        'UB',  # upper bound on external variable
        'LB',  # lower bound on external variable
    ],
)


@SolverFactory.register(
    'gdpopt.ldsda',
    doc="The LBB (logic-based branch and bound) Generalized Disjunctive "
    "Programming (GDP) solver",
)
class GDP_LDSDA_Solver(_GDPoptAlgorithm):
    """The GDPopt (Generalized Disjunctive Programming optimizer) logic-based
    branch and bound (LBB) solver.

    Accepts models that can include nonlinear, continuous variables and
    constraints, as well as logical conditions.
    """

    CONFIG = _GDPoptAlgorithm.CONFIG()
    _add_mip_solver_configs(CONFIG)
    _add_nlp_solver_configs(CONFIG, default_solver='ipopt')
    _add_nlp_solve_configs(
        CONFIG, default_nlp_init_method=restore_vars_to_original_values
    )
    _add_tolerance_configs(CONFIG)
    _add_ldsda_configs(CONFIG)

    algorithm = 'LDSDA'

    # Override solve() to customize the docstring for this solver
    @document_kwargs_from_configdict(CONFIG, doc=_GDPoptAlgorithm.solve.__doc__)
    def solve(self, model, **kwds):
        return super().solve(model, **kwds)

    def _log_citation(self, config):
        config.logger.info(
            "\n"
            + """- LDSDA algorithm:
        Bernal DE, Ovalle D, Liñán DA, Ricardez-Sandoval LA, Gómez JM, Grossmann IE.
        Process Superstructure Optimization through Discrete Steepest Descent Optimization: a GDP Analysis and Applications in Process Intensification.
        Computer Aided Chemical Engineering 2022 Jan 1 (Vol. 49, pp. 1279-1284). Elsevier.
        https://doi.org/10.1016/B978-0-323-85159-6.50213-X
        """.strip()
        )

    def _solve_gdp(self, model, config):
        logger = config.logger
        self.explored_nodes = 0

        # Create utility block on the original model so that we will be able to
        # copy solutions between
        util_block = self.original_util_block = add_util_block(model)
        add_disjunct_list(util_block)
        add_algebraic_variable_list(util_block)
        add_boolean_variable_lists(util_block)

        self.working_model = model.clone()
        # TODO: I don't like the name way, try something else?
        self.working_model_util_block = working_model_util_block = (
            self.working_model.component(util_block.name)
        )

        add_disjunction_list(working_model_util_block)
        # TODO: do we need to apply logical_to_disjunctive here?
        # This is applied in LBB.
        # root_node = TransformationFactory(
        #     'contrib.logical_to_disjunctive'
        # ).create_using(model)
        # Now that logical_to_disjunctive has been called.
        add_transformed_boolean_variable_list(working_model_util_block)

        self._log_header(logger)
        self.working_model_external_var_info_list = self.get_external_information(
            working_model_util_block, config
        )
        self.directions = self.get_directions(self.number_of_external_variables, config)
        self.best_direction = None
        self.current_point = config.starting_point
        self.explored_point_set = set()

        # Add the BigM suffix if it does not already exist. Used later during
        # nonlinear constraint activation.
        if not hasattr(working_model_util_block, 'BigM'):
            working_model_util_block.BigM = Suffix()

        locally_optimal = False
        # Solve the initial point
        self.fix_disjunctions_with_external_var(
            self.working_model_util_block, self.current_point
        )
        _ = self._solve_rnGDP_subproblem(self.working_model, config, 'Initial point')

        # Main loop
        while not locally_optimal:
            self.iteration += 1
            if self.any_termination_criterion_met(config):
                break
            locally_optimal = self.neighbor_search(self.working_model, config)
            if not locally_optimal:
                self.line_search(self.working_model, config)

        print("Optimal solution", self.current_point)

    def any_termination_criterion_met(self, config):
        return self.reached_iteration_limit(config) or self.reached_time_limit(config)

    def _solve_rnGDP_subproblem(self, model, config, search_type):
        subproblem = model.clone()
        TransformationFactory('core.logical_to_linear').apply_to(subproblem)
        TransformationFactory('gdp.bigm').apply_to(subproblem)

        try:
            with SuppressInfeasibleWarning():
                # TODO: we can use fbbt or deactivate trivial constraints here.
                # try:
                #     fbbt(subproblem, integer_tol=config.integer_tolerance)
                # except InfeasibleConstraintException:
                #     # copy variable values, even if errored
                #     copy_var_list_values(
                #         from_list=subprob_utils.algebraic_variable_list,
                #         to_list=model_utils.algebraic_variable_list,
                #         config=config,
                #         ignore_integrality=True,
                #     )
                #     return float('inf'), float('inf')
                minlp_args = dict(config.minlp_solver_args)
                if config.time_limit is not None and config.minlp_solver == 'gams':
                    elapsed = get_main_elapsed_time(self.timing)
                    remaining = max(config.time_limit - elapsed, 1)
                    minlp_args['add_options'] = minlp_args.get('add_options', [])
                    minlp_args['add_options'].append('option reslim=%s;' % remaining)
                result = SolverFactory(config.minlp_solver).solve(
                    subproblem, **minlp_args
                )
                primal_improved = self.handle_subproblem_result(
                    result, subproblem, config, search_type
                )
            return primal_improved
        except RuntimeError as e:
            config.logger.warning(
                "Solver encountered RuntimeError. Treating as infeasible. "
                "Msg: %s\n%s" % (str(e), traceback.format_exc())
            )
            return False

    def get_external_information(self, util_block, config):
        """Function that obtains information from the model to perform the reformulation with external variables.

        Parameters
        ----------
        util_block : Block
            The GDPOPT utility block of the model.
        config : ConfigBlock
            GDPopt configuration block

        Raises
        ------
        ValueError
            exactly_number is greater than 1
        """

        # self.working_model_util_block = []
        # util_block = self.working_model_util_block
        util_block.external_var_info_list = []
        model = util_block.parent_block()
        # Identify the variables that can be reformulated by performing a loop over logical constraints
        # TODO: we can automatically find all Exactly logical constraints in the model.
        # However, we cannot link the starting point and the logical constraint.
        # for c in util_block.logical_constraint_list:
        #     if isinstance(c.body, ExactlyExpression):
        for constraint_name in config.logical_constraint_list:
            # TODO: in the first version, we don't support more than one exactly constraint.
            # TODO: if we use component instead of model.find_component, it will fail.
            c = model.find_component(constraint_name)
            exactly_number = c.body.args[0]
            if exactly_number > 1:
                raise ValueError("The function only works for exactly_number = 1")
            sorted_boolean_var_list = sorted(c.body.args[1:], key=lambda x: x.index())
            util_block.external_var_info_list.append(
                ExternalVarInfo(
                    exactly_number=1,
                    Boolean_vars=sorted_boolean_var_list,
                    Disjuncts=[
                        boolean_var.get_associated_binary().parent_block()
                        for boolean_var in sorted_boolean_var_list
                    ],
                    LogicExpression=c.body,
                    UB=len(sorted_boolean_var_list),
                    LB=1,
                )
            )
        self.number_of_external_variables = sum(
            external_var_info.exactly_number
            for external_var_info in util_block.external_var_info_list
        )

    def fix_disjunctions_with_external_var(self, util_block, external_var_values_list):
        """Function that fixes the disjunctions in the model using the values of the external variables.

        Parameters
        ----------
        util_block : Block
            The GDPOPT utility block of the model.
        external_var_values_list : List
            The list of values of the external variables
        """
        for external_variable_value, external_var_info in zip(
            external_var_values_list, util_block.external_var_info_list
        ):
            for idx, (boolean_var, disjunct) in enumerate(
                zip(external_var_info.Boolean_vars, external_var_info.Disjuncts)
            ):
                if idx == external_variable_value - 1:
                    disjunct.activate()
                    boolean_var.fix(True)
                    disjunct.indicator_var.fix(True)
                    disjunct.binary_indicator_var.fix(1)
                else:
                    # TODO: maybe we can simplify this.
                    boolean_var.fix(False)
                    disjunct.indicator_var.fix(False)
                    disjunct.binary_indicator_var.fix(0)
                    disjunct.deactivate()
        self.explored_point_set.add(tuple(external_var_values_list))

    def get_directions(self, dimension, config):
        """Function creates the search directions of the given dimension.

        Parameters
        ----------
        dimension : int
            Dimension of the neighborhood
        config : ConfigBlock
            GDPopt configuration block

        Returns
        -------
        list
            the search directions.
        """
        if config.direction_norm == 'L2':
            directions = []
            for i in range(dimension):
                directions.append(tuple([0] * i + [1] + [0] * (dimension - i - 1)))
                directions.append(tuple([0] * i + [-1] + [0] * (dimension - i - 1)))
            return directions
        elif config.direction_norm == 'Linf':
            directions = list(it.product([-1, 0, 1], repeat=dimension))
            directions.remove((0,) * dimension)
            return directions

    def check_valid_neighbor(self, neighbor, external_var_info_list):
        """Function that checks if a given neighbor is valid.

        Parameters
        ----------
        neighbor : list
            the neighbor
        external_var_info_list : list
            the list of the external variable information

        Returns
        -------
        bool
            True if the neighbor is valid, False otherwise
        """
        if neighbor in self.explored_point_set:
            return False
        if all(
            external_var_value >= external_var_info.LB
            and external_var_value <= external_var_info.UB
            for external_var_value, external_var_info in zip(
                neighbor, external_var_info_list
            )
        ):
            return True
        else:
            return False

    def neighbor_search(self, model, config):
        """Function that evaluates a group of given points and returns the best

        Parameters
        ----------
        neighbor_list : list
            the list of neighbors
        model : ConcreteModel
            the subproblem model
        config : ConfigBlock
            GDPopt configuration block
        """
        locally_optimal = True
        best_neighbor = None
        # reset best direction
        self.best_direction = None
        for direction in self.directions:
            neighbor = tuple(map(sum, zip(self.current_point, direction)))
            if self.check_valid_neighbor(
                neighbor, self.working_model_util_block.external_var_info_list
            ):
                self.fix_disjunctions_with_external_var(
                    self.working_model_util_block, neighbor
                )
                primal_improved = self._solve_rnGDP_subproblem(
                    model, config, 'Neighbor search'
                )
                if primal_improved:
                    locally_optimal = False
                    best_neighbor = neighbor
                    self.best_direction = direction
        if not locally_optimal:
            self.current_point = best_neighbor
        return locally_optimal

    def line_search(self, model, config):
        """Function that performs a line search in a given direction.

        Parameters
        ----------
        model : ConcreteModel
            the subproblem model
        direction : list
            the direction
        config : ConfigBlock
            GDPopt configuration block
        """
        primal_improved = True
        while primal_improved:
            next_point = tuple(map(sum, zip(self.current_point, self.best_direction)))
            if not self.check_valid_neighbor(
                next_point, self.working_model_util_block.external_var_info_list
            ):
                break
            self.fix_disjunctions_with_external_var(
                self.working_model_util_block, next_point
            )
            primal_improved = self._solve_rnGDP_subproblem(model, config, 'Line search')
            # line_search_improved = self.handle_subproblem_result(
            #     subproblem_result, subproblem, config, 'Line search'
            # )
            if primal_improved:
                self.current_point = next_point
        print("Line search finished.")

    def handle_subproblem_result(
        self, subproblem_result, subproblem, config, search_type
    ):
        """Function that handles the result of the subproblem

        Parameters
        ----------
        subproblem : ConcreteModel
            the subproblem model
        subproblem_result : tuple
            the result of the subproblem
        config : ConfigBlock
            GDPopt configuration block

        Returns
        -------
        bool
            True if the result improved the current point, False otherwise
        """
        if subproblem_result is None:
            return False
        if subproblem_result.solver.termination_condition in {
            tc.optimal,
            tc.feasible,
            tc.globallyOptimal,
            tc.locallyOptimal,
            tc.maxTimeLimit,
            tc.maxIterations,
            tc.maxEvaluations,
        }:
            primal_bound = (
                subproblem_result.problem.upper_bound
                if self.objective_sense == minimize
                else subproblem_result.problem.lower_bound
            )
            primal_improved = self._update_bounds_after_solve(
                search_type, primal=primal_bound, logger=config.logger
            )
            if primal_improved:
                self.update_incumbent(
                    subproblem.component(self.original_util_block.name)
                )
            return primal_improved
        return False

    def _log_header(self, logger):
        logger.info(
            '================================================================='
            '============================'
        )
        logger.info(
            '{:^9} | {:^15} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration',
                'Search Type',
                'Lower Bound',
                'Upper Bound',
                ' Gap ',
                'Time(s)',
            )
        )
