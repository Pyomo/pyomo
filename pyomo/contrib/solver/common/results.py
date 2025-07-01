#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
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
    DEVELOPER_OPTION,
)
from pyomo.opt.results.solution import SolutionStatus as LegacySolutionStatus
from pyomo.opt.results.solver import (
    TerminationCondition as LegacyTerminationCondition,
    SolverStatus as LegacySolverStatus,
)


class TerminationCondition(enum.Enum):
    """
    An Enum that enumerates all possible exit statuses for a solver call.

    """

    convergenceCriteriaSatisfied = 0
    "The solver exited because convergence criteria of the problem were satisfied."

    maxTimeLimit = 1
    """The solver exited due to reaching a specified time limit."""

    iterationLimit = 2
    """The solver exited due to reaching a specified iteration limit."""

    objectiveLimit = 3
    """The solver exited due to reaching an objective limit. For example, in
    Gurobi, the exit message "Optimal objective for model was proven to
    be worse than the value specified in the Cutoff parameter" would map
    to objectiveLimit.
    """

    minStepLength = 4
    """The solver exited due to a minimum step length.  Minimum step length
    reached may mean that the problem is infeasible or that the problem
    is feasible but the solver could not converge.
    """

    unbounded = 5
    "The solver exited because the problem has been found to be unbounded."

    provenInfeasible = 6
    "The solver exited because the problem has been proven infeasible."

    locallyInfeasible = 7
    """The solver exited because no feasible solution was found to the
    submitted problem, but it could not be proven that no such solution
    exists.
    """

    infeasibleOrUnbounded = 8
    """Some solvers do not specify between infeasibility or unboundedness
    and instead return that one or the other has occurred. For example,
    in Gurobi, this may occur because there are some steps in presolve
    that prevent Gurobi from distinguishing between infeasibility and
    unboundedness.
    """

    error = 9
    """The solver exited with some error. The error message will also be
    captured and returned.
    """

    interrupted = 10
    "The solver was interrupted while running."

    licensingProblems = 11
    """The solver experienced issues with licensing. This could be that no
    license was found, the license is of the wrong type for the problem
    (e.g., problem is too big for type of license), or there was an
    issue contacting a licensing server.
    """

    emptyModel = 12
    "The model being solved did not have any variables"

    unknown = 42
    "All other unrecognized exit statuses fall in this category."


class SolutionStatus(enum.Enum):
    """An enumeration for interpreting the result of a termination. This
    describes the designated status by the solver to be loaded back into
    the model.

    """

    noSolution = 0
    """No (single) solution was found; possible that a population of
    solutions was returned.
    """

    infeasible = 10
    "Solution point does not satisfy some domains and/or constraints."

    feasible = 20
    "A solution for which all of the constraints in the model are satisfied."

    optimal = 30
    "A feasible solution satisfying the solver's optimality criteria."


class Results(ConfigDict):
    """Base class for all solver results"""

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
                description="Object for loading the solution back into the model.",
                visibility=DEVELOPER_OPTION,
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
        self.solver_config: ConfigDict = self.declare(
            'solver_config',
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


# The logic for the new solution status to legacy solution status
# cannot be contained in a dictionary because it is a many -> one
# relationship.
def legacy_solution_status_map(results):
    """
    Map the new TerminationCondition/SolutionStatus values into LegacySolutionStatus
    objects. Because we condensed results objects, some of the previous statuses
    are no longer clearly achievable.
    """
    if results.termination_condition in set(
        [
            TerminationCondition.maxTimeLimit,
            TerminationCondition.iterationLimit,
            TerminationCondition.objectiveLimit,
            TerminationCondition.minStepLength,
        ]
    ):
        return LegacySolutionStatus.stoppedByLimit
    if results.termination_condition in set(
        [TerminationCondition.provenInfeasible, TerminationCondition.locallyInfeasible]
    ):
        return LegacySolutionStatus.infeasible
    if results.termination_condition in set(
        [
            TerminationCondition.error,
            TerminationCondition.licensingProblems,
            TerminationCondition.interrupted,
        ]
    ):
        return LegacySolutionStatus.error
    if results.termination_condition == TerminationCondition.unbounded:
        return LegacySolutionStatus.unbounded
    if (
        results.termination_condition
        == TerminationCondition.convergenceCriteriaSatisfied
    ):
        if results.solution_status == SolutionStatus.feasible:
            return LegacySolutionStatus.feasible
        return LegacySolutionStatus.optimal
    return LegacySolutionStatus.unknown
