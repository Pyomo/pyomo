#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2024
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import enum
from typing import Optional, Tuple
from datetime import datetime

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    IsInstance,
    NonNegativeInt,
    In,
    NonNegativeFloat,
    ADVANCED_OPTION,
)
from pyomo.opt.results.solution import SolutionStatus as LegacySolutionStatus
from pyomo.opt.results.solver import (
    TerminationCondition as LegacyTerminationCondition,
    SolverStatus as LegacySolverStatus,
)


class TerminationCondition(enum.Enum):
    """
    An Enum that enumerates all possible exit statuses for a solver call.

    Attributes
    ----------
    convergenceCriteriaSatisfied: 0
        The solver exited because convergence criteria of the problem were
        satisfied.
    maxTimeLimit: 1
        The solver exited due to reaching a specified time limit.
    iterationLimit: 2
        The solver exited due to reaching a specified iteration limit.
    objectiveLimit: 3
        The solver exited due to reaching an objective limit. For example,
        in Gurobi, the exit message "Optimal objective for model was proven to
        be worse than the value specified in the Cutoff parameter" would map
        to objectiveLimit.
    minStepLength: 4
        The solver exited due to a minimum step length.
        Minimum step length reached may mean that the problem is infeasible or
        that the problem is feasible but the solver could not converge.
    unbounded: 5
        The solver exited because the problem has been found to be unbounded.
    provenInfeasible: 6
        The solver exited because the problem has been proven infeasible.
    locallyInfeasible: 7
        The solver exited because no feasible solution was found to the
        submitted problem, but it could not be proven that no such solution exists.
    infeasibleOrUnbounded: 8
        Some solvers do not specify between infeasibility or unboundedness and
        instead return that one or the other has occurred. For example, in
        Gurobi, this may occur because there are some steps in presolve that
        prevent Gurobi from distinguishing between infeasibility and unboundedness.
    error: 9
        The solver exited with some error. The error message will also be
        captured and returned.
    interrupted: 10
        The solver was interrupted while running.
    licensingProblems: 11
        The solver experienced issues with licensing. This could be that no
        license was found, the license is of the wrong type for the problem (e.g.,
        problem is too big for type of license), or there was an issue contacting
        a licensing server.
    emptyModel: 12
        The model being solved did not have any variables
    unknown: 42
        All other unrecognized exit statuses fall in this category.
    """

    convergenceCriteriaSatisfied = 0

    maxTimeLimit = 1

    iterationLimit = 2

    objectiveLimit = 3

    minStepLength = 4

    unbounded = 5

    provenInfeasible = 6

    locallyInfeasible = 7

    infeasibleOrUnbounded = 8

    error = 9

    interrupted = 10

    licensingProblems = 11

    emptyModel = 12

    unknown = 42


class SolutionStatus(enum.Enum):
    """
    An enumeration for interpreting the result of a termination. This describes the designated
    status by the solver to be loaded back into the model.

    Attributes
    ----------
    noSolution: 0
        No (single) solution was found; possible that a population of solutions
        was returned.
    infeasible: 10
        Solution point does not satisfy some domains and/or constraints.
    feasible: 20
        A solution for which all of the constraints in the model are satisfied.
    optimal: 30
        A feasible solution where the objective function reaches its specified
        sense (e.g., maximum, minimum)
    """

    noSolution = 0

    infeasible = 10

    feasible = 20

    optimal = 30


class Results(ConfigDict):
    """
    Attributes
    ----------
    solution_loader: SolutionLoaderBase
        Object for loading the solution back into the model.
    termination_condition: :class:`TerminationCondition<pyomo.contrib.solver.results.TerminationCondition>`
        The reason the solver exited. This is a member of the
        TerminationCondition enum.
    solution_status: :class:`SolutionStatus<pyomo.contrib.solver.results.SolutionStatus>`
        The result of the solve call. This is a member of the SolutionStatus
        enum.
    incumbent_objective: float
        If a feasible solution was found, this is the objective value of
        the best solution found. If no feasible solution was found, this is
        None.
    objective_bound: float
        The best objective bound found. For minimization problems, this is
        the lower bound. For maximization problems, this is the upper bound.
        For solvers that do not provide an objective bound, this should be -inf
        (minimization) or inf (maximization)
    solver_name: str
        The name of the solver in use.
    solver_version: tuple
        A tuple representing the version of the solver in use.
    iteration_count: int
        The total number of iterations.
    timing_info: ConfigDict
        A ConfigDict containing three pieces of information:
            - ``start_timestamp``: UTC timestamp of when run was initiated
            - ``wall_time``: elapsed wall clock time for entire process
            - ``timer``: a HierarchicalTimer object containing timing data about the solve

        Specific solvers may add other relevant timing information, as appropriate.
    extra_info: ConfigDict
        A ConfigDict to store extra information such as solver messages.
    solver_configuration: ConfigDict
        A copy of the SolverConfig ConfigDict, for later inspection/reproducibility.
    solver_log: str
        (ADVANCED OPTION) Any solver log messages.
    """

    def __init__(
        self,
        description=None,
        doc=None,
        implicit=False,
        implicit_domain=None,
        visibility=0,
    ):
        super().__init__(
            description=description,
            doc=doc,
            implicit=implicit,
            implicit_domain=implicit_domain,
            visibility=visibility,
        )

        self.solution_loader = self.declare(
            'solution_loader',
            ConfigValue(
                description="Object for loading the solution back into the model."
            ),
        )
        self.termination_condition: TerminationCondition = self.declare(
            'termination_condition',
            ConfigValue(
                domain=In(TerminationCondition),
                default=TerminationCondition.unknown,
                description="The reason the solver exited. This is a member of the "
                "TerminationCondition enum.",
            ),
        )
        self.solution_status: SolutionStatus = self.declare(
            'solution_status',
            ConfigValue(
                domain=In(SolutionStatus),
                default=SolutionStatus.noSolution,
                description="The result of the solve call. This is a member of "
                "the SolutionStatus enum.",
            ),
        )
        self.incumbent_objective: Optional[float] = self.declare(
            'incumbent_objective',
            ConfigValue(
                domain=float,
                default=None,
                description="If a feasible solution was found, this is the objective "
                "value of the best solution found. If no feasible solution was found, this is None.",
            ),
        )
        self.objective_bound: Optional[float] = self.declare(
            'objective_bound',
            ConfigValue(
                domain=float,
                default=None,
                description="The best objective bound found. For minimization problems, "
                "this is the lower bound. For maximization problems, this is the "
                "upper bound. For solvers that do not provide an objective bound, "
                "this should be -inf (minimization) or inf (maximization)",
            ),
        )
        self.solver_name: Optional[str] = self.declare(
            'solver_name',
            ConfigValue(domain=str, description="The name of the solver in use."),
        )
        self.solver_version: Optional[Tuple[int, ...]] = self.declare(
            'solver_version',
            ConfigValue(
                domain=tuple,
                description="A tuple representing the version of the solver in use.",
            ),
        )
        self.iteration_count: Optional[int] = self.declare(
            'iteration_count',
            ConfigValue(
                domain=NonNegativeInt,
                default=None,
                description="The total number of iterations.",
            ),
        )
        self.timing_info: ConfigDict = self.declare(
            'timing_info', ConfigDict(implicit=True)
        )

        self.timing_info.start_timestamp: datetime = self.timing_info.declare(
            'start_timestamp',
            ConfigValue(
                domain=IsInstance(datetime),
                description="UTC timestamp of when run was initiated.",
            ),
        )
        self.timing_info.wall_time: Optional[float] = self.timing_info.declare(
            'wall_time',
            ConfigValue(
                domain=NonNegativeFloat,
                description="Elapsed wall clock time for entire process.",
            ),
        )
        self.extra_info: ConfigDict = self.declare(
            'extra_info', ConfigDict(implicit=True)
        )
        self.solver_configuration: ConfigDict = self.declare(
            'solver_configuration',
            ConfigValue(
                description="A copy of the config object used in the solve call.",
                visibility=ADVANCED_OPTION,
            ),
        )
        self.solver_log: str = self.declare(
            'solver_log',
            ConfigValue(
                domain=str,
                default=None,
                visibility=ADVANCED_OPTION,
                description="Any solver log messages.",
            ),
        )

    def display(
        self, content_filter=None, indent_spacing=2, ostream=None, visibility=0
    ):
        return super().display(content_filter, indent_spacing, ostream, visibility)


# Everything below here preserves backwards compatibility

legacy_termination_condition_map = {
    TerminationCondition.unknown: LegacyTerminationCondition.unknown,
    TerminationCondition.maxTimeLimit: LegacyTerminationCondition.maxTimeLimit,
    TerminationCondition.iterationLimit: LegacyTerminationCondition.maxIterations,
    TerminationCondition.objectiveLimit: LegacyTerminationCondition.minFunctionValue,
    TerminationCondition.minStepLength: LegacyTerminationCondition.minStepLength,
    TerminationCondition.convergenceCriteriaSatisfied: LegacyTerminationCondition.optimal,
    TerminationCondition.unbounded: LegacyTerminationCondition.unbounded,
    TerminationCondition.provenInfeasible: LegacyTerminationCondition.infeasible,
    TerminationCondition.locallyInfeasible: LegacyTerminationCondition.infeasible,
    TerminationCondition.infeasibleOrUnbounded: LegacyTerminationCondition.infeasibleOrUnbounded,
    TerminationCondition.error: LegacyTerminationCondition.error,
    TerminationCondition.interrupted: LegacyTerminationCondition.resourceInterrupt,
    TerminationCondition.licensingProblems: LegacyTerminationCondition.licensingProblems,
}


legacy_solver_status_map = {
    TerminationCondition.unknown: LegacySolverStatus.unknown,
    TerminationCondition.maxTimeLimit: LegacySolverStatus.aborted,
    TerminationCondition.iterationLimit: LegacySolverStatus.aborted,
    TerminationCondition.objectiveLimit: LegacySolverStatus.aborted,
    TerminationCondition.minStepLength: LegacySolverStatus.error,
    TerminationCondition.convergenceCriteriaSatisfied: LegacySolverStatus.ok,
    TerminationCondition.unbounded: LegacySolverStatus.error,
    TerminationCondition.provenInfeasible: LegacySolverStatus.error,
    TerminationCondition.locallyInfeasible: LegacySolverStatus.error,
    TerminationCondition.infeasibleOrUnbounded: LegacySolverStatus.error,
    TerminationCondition.error: LegacySolverStatus.error,
    TerminationCondition.interrupted: LegacySolverStatus.aborted,
    TerminationCondition.licensingProblems: LegacySolverStatus.error,
}


legacy_solution_status_map = {
    SolutionStatus.noSolution: LegacySolutionStatus.unknown,
    SolutionStatus.noSolution: LegacySolutionStatus.stoppedByLimit,
    SolutionStatus.noSolution: LegacySolutionStatus.error,
    SolutionStatus.noSolution: LegacySolutionStatus.other,
    SolutionStatus.noSolution: LegacySolutionStatus.unsure,
    SolutionStatus.noSolution: LegacySolutionStatus.unbounded,
    SolutionStatus.optimal: LegacySolutionStatus.locallyOptimal,
    SolutionStatus.optimal: LegacySolutionStatus.globallyOptimal,
    SolutionStatus.optimal: LegacySolutionStatus.optimal,
    SolutionStatus.infeasible: LegacySolutionStatus.infeasible,
    SolutionStatus.feasible: LegacySolutionStatus.feasible,
    SolutionStatus.feasible: LegacySolutionStatus.bestSoFar,
}
