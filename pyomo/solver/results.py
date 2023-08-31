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

import enum
from typing import Optional

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    NonNegativeInt,
    In,
    NonNegativeFloat,
)
from pyomo.opt.results.solution import SolutionStatus as LegacySolutionStatus
from pyomo.opt.results.solver import (
    TerminationCondition as LegacyTerminationCondition,
    SolverStatus as LegacySolverStatus,
)


class TerminationCondition(enum.Enum):
    """
    An enumeration for checking the termination condition of solvers
    """

    """unknown serves as both a default value, and it is used when no other enum member makes sense"""
    unknown = 42

    """The solver exited because the convergence criteria were satisfied"""
    convergenceCriteriaSatisfied = 0

    """The solver exited due to a time limit"""
    maxTimeLimit = 1

    """The solver exited due to an iteration limit"""
    iterationLimit = 2

    """The solver exited due to an objective limit"""
    objectiveLimit = 3

    """The solver exited due to a minimum step length"""
    minStepLength = 4

    """The solver exited because the problem is unbounded"""
    unbounded = 5

    """The solver exited because the problem is proven infeasible"""
    provenInfeasible = 6

    """The solver exited because the problem was found to be locally infeasible"""
    locallyInfeasible = 7

    """The solver exited because the problem is either infeasible or unbounded"""
    infeasibleOrUnbounded = 8

    """The solver exited due to an error"""
    error = 9

    """The solver exited because it was interrupted"""
    interrupted = 10

    """The solver exited due to licensing problems"""
    licensingProblems = 11


class SolutionStatus(enum.IntEnum):
    """
    An enumeration for interpreting the result of a termination. This describes the designated
    status by the solver to be loaded back into the model.

    For now, we are choosing to use IntEnum such that return values are numerically
    assigned in increasing order.
    """

    """No (single) solution found; possible that a population of solutions was returned"""
    noSolution = 0

    """Solution point does not satisfy some domains and/or constraints"""
    infeasible = 10

    """Feasible solution identified"""
    feasible = 20

    """Optimal solution identified"""
    optimal = 30


class Results(ConfigDict):
    """
    Attributes
    ----------
    termination_condition: TerminationCondition
        The reason the solver exited. This is a member of the
        TerminationCondition enum.
    incumbent_objective: float
        If a feasible solution was found, this is the objective value of
        the best solution found. If no feasible solution was found, this is
        None.
    objective_bound: float
        The best objective bound found. For minimization problems, this is
        the lower bound. For maximization problems, this is the upper bound.
        For solvers that do not provide an objective bound, this should be -inf
        (minimization) or inf (maximization)
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

        self.solution_loader = self.declare('solution_loader', ConfigValue())
        self.termination_condition: In(TerminationCondition) = self.declare(
            'termination_condition',
            ConfigValue(
                domain=In(TerminationCondition), default=TerminationCondition.unknown
            ),
        )
        self.solution_status: In(SolutionStatus) = self.declare(
            'solution_status',
            ConfigValue(domain=In(SolutionStatus), default=SolutionStatus.noSolution),
        )
        self.incumbent_objective: Optional[float] = self.declare(
            'incumbent_objective', ConfigValue(domain=float)
        )
        self.objective_bound: Optional[float] = self.declare(
            'objective_bound', ConfigValue(domain=float)
        )
        self.solver_name: Optional[str] = self.declare(
            'solver_name', ConfigValue(domain=str)
        )
        self.solver_version: Optional[tuple] = self.declare(
            'solver_version', ConfigValue(domain=tuple)
        )
        self.iteration_count: NonNegativeInt = self.declare(
            'iteration_count', ConfigValue(domain=NonNegativeInt)
        )
        self.timing_info: ConfigDict = self.declare('timing_info', ConfigDict())
        self.timing_info.start_time = self.timing_info.declare(
            'start_time', ConfigValue()
        )
        self.timing_info.wall_time: NonNegativeFloat = self.timing_info.declare(
            'wall_time', ConfigValue(domain=NonNegativeFloat)
        )
        self.timing_info.solver_wall_time: NonNegativeFloat = self.timing_info.declare(
            'solver_wall_time', ConfigValue(domain=NonNegativeFloat)
        )
        self.extra_info: ConfigDict = self.declare(
            'extra_info', ConfigDict(implicit=True)
        )

    def __str__(self):
        s = ''
        s += 'termination_condition: ' + str(self.termination_condition) + '\n'
        s += 'solution_status: ' + str(self.solution_status) + '\n'
        s += 'incumbent_objective: ' + str(self.incumbent_objective) + '\n'
        s += 'objective_bound: ' + str(self.objective_bound)
        return s


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
