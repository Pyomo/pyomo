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
import traceback
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
    _add_ldsda_configs,
    _add_mip_solver_configs,
    _add_tolerance_configs,
    _add_nlp_solve_configs,
)
from pyomo.contrib.gdpopt.nlp_initialization import restore_vars_to_original_values
from pyomo.contrib.gdpopt.util import SuppressInfeasibleWarning, get_main_elapsed_time
from pyomo.contrib.satsolver.satsolver import satisfiable
from pyomo.core import minimize, Suffix, TransformationFactory
from pyomo.opt import SolverFactory
from pyomo.opt import TerminationCondition as tc
from pyomo.core.expr.logical_expr import ExactlyExpression
from pyomo.common.dependencies import attempt_import

it, it_available = attempt_import('itertools')
tabulate, tabulate_available = attempt_import('tabulate')

# Data tuple for external variables.
ExternalVarInfo = namedtuple(
    'ExternalVarInfo',
    [
        'exactly_number',  # number of external variables for this type
        'Boolean_vars',  # list with names of the ordered Boolean variables to be reformulated
        'UB',  # upper bound on external variable
        'LB',  # lower bound on external variable
    ],
)


@SolverFactory.register(
    'gdpopt.ldsda',
    doc="The LD-SDA (Logic-based Discrete-Steepest Descent Algorithm)"
    "Generalized Disjunctive Programming (GDP) solver",
)
class GDP_LDSDA_Solver(_GDPoptAlgorithm):
    """The GDPopt (Generalized Disjunctive Programming optimizer)
    LD-SDA (Logic-based Discrete-Steepest Descent (LD-SDA) solver.

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
        """Solve the GDP model.

        Parameters
        ----------
        model : ConcreteModel
            The GDP model to be solved
        config : ConfigBlock
            GDPopt configuration block
        """
        logger = config.logger
        self.log_formatter = (
            '{:>9}   {:>15}   {:>20}   {:>11.5f}   {:>11.5f}   {:>8.2%}   {:>7.2f}  {}'
        )
        self.best_direction = None
        self.current_point = tuple(config.starting_point)
        self.explored_point_set = set()

        # Create utility block on the original model so that we will be able to
        # copy solutions between
        util_block = self.original_util_block = add_util_block(model)
        add_disjunct_list(util_block)
        add_algebraic_variable_list(util_block)
        add_boolean_variable_lists(util_block)

        # We will use the working_model to perform the LDSDA search.
        self.working_model = model.clone()
        # TODO: I don't like the name way, try something else?
        self.working_model_util_block = self.working_model.component(util_block.name)

        add_disjunction_list(self.working_model_util_block)
        TransformationFactory('core.logical_to_linear').apply_to(self.working_model)
        # Now that logical_to_disjunctive has been called.
        add_transformed_boolean_variable_list(self.working_model_util_block)
        self._get_external_information(self.working_model_util_block, config)
        self.directions = self._get_directions(
            self.number_of_external_variables, config
        )

        # Add the BigM suffix if it does not already exist. Used later during
        # nonlinear constraint activation.
        if not hasattr(self.working_model_util_block, 'BigM'):
            self.working_model_util_block.BigM = Suffix()
        self._log_header(logger)
        # Solve the initial point
        _ = self._solve_GDP_subproblem(self.current_point, 'Initial point', config)

        # Main loop
        locally_optimal = False
        while not locally_optimal:
            self.iteration += 1
            if self.any_termination_criterion_met(config):
                break
            locally_optimal = self.neighbor_search(config)
            if not locally_optimal:
                self.line_search(config)

    def any_termination_criterion_met(self, config):
        return self.reached_iteration_limit(config) or self.reached_time_limit(config)

    def _solve_GDP_subproblem(self, external_var_value, search_type, config):
        """Solve the GDP subproblem with disjunctions fixed according to the external variable.

        Parameters
        ----------
        external_var_value : list
            The values of the external variables to be evaluated
        search_type : str
            The type of search, neighbor search or line search
        config : ConfigBlock
            GDPopt configuration block

        Returns
        -------
        bool
            weather the primal bound is improved
        """
        self.fix_disjunctions_with_external_var(external_var_value)
        subproblem = self.working_model.clone()
        TransformationFactory('core.logical_to_linear').apply_to(subproblem)

        try:
            with SuppressInfeasibleWarning():
                try:
                    TransformationFactory('gdp.bigm').apply_to(subproblem)
                    fbbt(subproblem, integer_tol=config.integer_tolerance)
                    TransformationFactory('contrib.detect_fixed_vars').apply_to(
                        subproblem
                    )
                    TransformationFactory('contrib.propagate_fixed_vars').apply_to(
                        subproblem
                    )
                    TransformationFactory(
                        'contrib.deactivate_trivial_constraints'
                    ).apply_to(subproblem, tmp=False, ignore_infeasible=False)
                except InfeasibleConstraintException:
                    return False
                minlp_args = dict(config.minlp_solver_args)
                if config.time_limit is not None and config.minlp_solver == 'gams':
                    elapsed = get_main_elapsed_time(self.timing)
                    remaining = max(config.time_limit - elapsed, 1)
                    minlp_args['add_options'] = minlp_args.get('add_options', [])
                    minlp_args['add_options'].append('option reslim=%s;' % remaining)
                result = SolverFactory(config.minlp_solver).solve(
                    subproblem, **minlp_args
                )
                primal_improved = self._handle_subproblem_result(
                    result, subproblem, external_var_value, config, search_type
                )
            return primal_improved
        except RuntimeError as e:
            config.logger.warning(
                "Solver encountered RuntimeError. Treating as infeasible. "
                "Msg: %s\n%s" % (str(e), traceback.format_exc())
            )
            return False

    def _get_external_information(self, util_block, config):
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
        util_block.external_var_info_list = []
        model = util_block.parent_block()
        reformulation_summary = []
        # Identify the variables that can be reformulated by performing a loop over logical constraints
        # TODO: we can automatically find all Exactly logical constraints in the model.
        # However, we cannot link the starting point and the logical constraint.
        # for c in util_block.logical_constraint_list:
        #     if isinstance(c.body, ExactlyExpression):
        if config.logical_constraint_list is not None:
            for constraint_name in config.logical_constraint_list:
                # TODO: in the first version, we don't support more than one exactly constraint.
                # TODO: if we use component instead of model.find_component, it will fail.
                c = model.find_component(constraint_name)
                exactly_number = c.body.args[0]
                if exactly_number > 1:
                    raise ValueError("The function only works for exactly_number = 1")
                sorted_boolean_var_list = c.body.args[1:]
                util_block.external_var_info_list.append(
                    ExternalVarInfo(
                        exactly_number=1,
                        Boolean_vars=sorted_boolean_var_list,
                        UB=len(sorted_boolean_var_list),
                        LB=1,
                    )
                )
                reformulation_summary.append(
                    [
                        1,
                        len(sorted_boolean_var_list),
                        [boolean_var.name for boolean_var in sorted_boolean_var_list],
                    ]
                )
        if config.disjunction_list is not None:
            for disjunction_name in config.disjunction_list:
                # TODO: in the first version, we don't support more than one exactly constraint.
                # TODO: if we use component instead of model.find_component, it will fail.
                disjunction = model.find_component(disjunction_name)
                sorted_boolean_var_list = [
                    disjunct.indicator_var for disjunct in disjunction.disjuncts
                ]
                util_block.external_var_info_list.append(
                    ExternalVarInfo(
                        exactly_number=1,
                        Boolean_vars=sorted_boolean_var_list,
                        UB=len(sorted_boolean_var_list),
                        LB=1,
                    )
                )
                reformulation_summary.append(
                    [
                        1,
                        len(sorted_boolean_var_list),
                        [boolean_var.name for boolean_var in sorted_boolean_var_list],
                    ]
                )
        config.logger.info("Reformulation Summary:")
        config.logger.info(
            tabulate.tabulate(
                reformulation_summary,
                headers=["Ext Var Index", "LB", "UB", "Associated Boolean Vars"],
                showindex="always",
                tablefmt="simple_outline",
            )
        )
        self.number_of_external_variables = sum(
            external_var_info.exactly_number
            for external_var_info in util_block.external_var_info_list
        )
        if self.number_of_external_variables != len(config.starting_point):
            raise ValueError(
                "The length of the provided starting point doesn't equal to the number of disjunctions."
            )

    def fix_disjunctions_with_external_var(self, external_var_values_list):
        """Function that fixes the disjunctions in the working_model using the values of the external variables.

        Parameters
        ----------
        external_var_values_list : List
            The list of values of the external variables
        """
        for external_variable_value, external_var_info in zip(
            external_var_values_list,
            self.working_model_util_block.external_var_info_list,
        ):
            for idx, boolean_var in enumerate(external_var_info.Boolean_vars):
                if idx == external_variable_value - 1:
                    boolean_var.fix(True)
                    if boolean_var.get_associated_binary() is not None:
                        boolean_var.get_associated_binary().fix(1)
                else:
                    boolean_var.fix(False)
                    if boolean_var.get_associated_binary() is not None:
                        boolean_var.get_associated_binary().fix(0)
        self.explored_point_set.add(tuple(external_var_values_list))

    def _get_directions(self, dimension, config):
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
        else:
            raise ValueError(
                "The direction_norm option must be 'L2' or 'Linf', "
                "but received %s" % config.direction_norm
            )

    def _check_valid_neighbor(self, neighbor):
        """Function that checks if a given neighbor is valid.

        Parameters
        ----------
        neighbor : list
            the neighbor to be checked

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
                neighbor, self.working_model_util_block.external_var_info_list
            )
        ):
            return True
        else:
            return False

    def neighbor_search(self, config):
        """Function that evaluates a group of given points and returns the best

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block
        """
        locally_optimal = True
        best_neighbor = None
        self.best_direction = None  # reset best direction
        for direction in self.directions:
            neighbor = tuple(map(sum, zip(self.current_point, direction)))
            if self._check_valid_neighbor(neighbor):
                primal_improved = self._solve_GDP_subproblem(
                    neighbor, 'Neighbor search', config
                )
                if primal_improved:
                    locally_optimal = False
                    best_neighbor = neighbor
                    self.best_direction = direction
        if not locally_optimal:
            self.current_point = best_neighbor
        return locally_optimal

    def line_search(self, config):
        """Function that performs a line search in the best direction.

        Parameters
        ----------
        config : ConfigBlock
            GDPopt configuration block
        """
        primal_improved = True
        while primal_improved:
            next_point = tuple(map(sum, zip(self.current_point, self.best_direction)))
            if self._check_valid_neighbor(next_point):
                primal_improved = self._solve_GDP_subproblem(
                    next_point, 'Line search', config
                )
                if primal_improved:
                    self.current_point = next_point
            else:
                break

    def _handle_subproblem_result(
        self, subproblem_result, subproblem, external_var_value, config, search_type
    ):
        """Function that handles the result of the subproblem

        Parameters
        ----------
        subproblem_result : tuple
            the result of the subproblem
        subproblem : ConcreteModel
            the subproblem model
        external_var_value : list
            the values of the external variables
        config : ConfigBlock
            GDPopt configuration block
        search_type : str
            the type of search, neighbor search or line search

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
                search_type,
                primal=primal_bound,
                logger=config.logger,
                current_point=external_var_value,
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
            '===================================='
        )
        logger.info(
            '{:^9} | {:^15} | {:^20} | {:^11} | {:^11} | {:^8} | {:^7}\n'.format(
                'Iteration',
                'Search Type',
                'External Variables',
                'Lower Bound',
                'Upper Bound',
                ' Gap ',
                'Time(s)',
            )
        )

    def _log_current_state(
        self, logger, search_type, current_point, primal_improved=False
    ):
        star = "*" if primal_improved else ""
        logger.info(
            self.log_formatter.format(
                self.iteration,
                search_type,
                str(current_point),
                self.LB,
                self.UB,
                self.relative_gap(),
                get_main_elapsed_time(self.timing),
                star,
            )
        )

    def _update_bounds_after_solve(
        self, search_type, primal=None, dual=None, logger=None, current_point=None
    ):
        primal_improved = self._update_bounds(primal, dual)
        if logger is not None:
            self._log_current_state(logger, search_type, current_point, primal_improved)

        return primal_improved
