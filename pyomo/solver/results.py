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
from typing import Optional, Tuple
from datetime import datetime

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
from pyomo.solver.solution import SolutionLoaderBase
from pyomo.solver.util import SolverSystemError


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

    unknown = 42


class SolutionStatus(enum.IntEnum):
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
    termination_condition: TerminationCondition
        The reason the solver exited. This is a member of the
        TerminationCondition enum.
    solution_status: SolutionStatus
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
            start_time: UTC timestamp of when run was initiated
            wall_time: elapsed wall clock time for entire process
            solver_wall_time: elapsed wall clock time for solve call
    extra_info: ConfigDict
        A ConfigDict to store extra information such as solver messages.
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

        self.solution_loader: SolutionLoaderBase = self.declare(
            'solution_loader', ConfigValue()
        )
        self.termination_condition: TerminationCondition = self.declare(
            'termination_condition',
            ConfigValue(
                domain=In(TerminationCondition), default=TerminationCondition.unknown
            ),
        )
        self.solution_status: SolutionStatus = self.declare(
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
        self.solver_version: Optional[Tuple[int, ...]] = self.declare(
            'solver_version', ConfigValue(domain=tuple)
        )
        self.iteration_count: Optional[int] = self.declare(
            'iteration_count', ConfigValue(domain=NonNegativeInt)
        )
        self.timing_info: ConfigDict = self.declare('timing_info', ConfigDict())
        # TODO: Implement type checking for datetime
        self.timing_info.start_time: datetime = self.timing_info.declare(
            'start_time', ConfigValue()
        )
        self.timing_info.wall_time: Optional[float] = self.timing_info.declare(
            'wall_time', ConfigValue(domain=NonNegativeFloat)
        )
        self.timing_info.solver_wall_time: Optional[float] = self.timing_info.declare(
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


class ResultsReader:
    pass


def parse_sol_file(file, results):
    # The original reader for sol files is in pyomo.opt.plugins.sol.
    # Per my original complaint, it has "magic numbers" that I just don't
    # know how to test. It's apparently less fragile than that in APPSI.
    # NOTE: The Results object now also holds the solution loader, so we do
    # not need pass in a solution like we did previously.
    if results is None:
        results = Results()

    # For backwards compatibility and general safety, we will parse all
    # lines until "Options" appears. Anything before "Options" we will
    # consider to be the solver message.
    message = []
    for line in file:
        if not line:
            break
        line = line.strip()
        if "Options" in line:
            break
        message.append(line)
    message = '\n'.join(message)
    # Once "Options" appears, we must now read the content under it.
    model_objects = []
    if "Options" in line:
        line = file.readline()
        number_of_options = int(line)
        need_tolerance = False
        if number_of_options > 4: # MRM: Entirely unclear why this is necessary, or if it even is
            number_of_options -= 2
            need_tolerance = True
        for i in range(number_of_options + 4):
            line = file.readline()
            model_objects.append(int(line))
        if need_tolerance: # MRM: Entirely unclear why this is necessary, or if it even is
            line = file.readline()
            model_objects.append(float(line))
    else:
        raise SolverSystemError("ERROR READING `sol` FILE. No 'Options' line found.")
    # Identify the total number of variables and constraints
    number_of_cons = model_objects[number_of_options + 1]
    number_of_vars = model_objects[number_of_options + 3]
    constraints = []
    variables = []
    # Parse through the constraint lines and capture the constraints
    i = 0
    while i < number_of_cons:
        line = file.readline()
        constraints.append(float(line))
    # Parse through the variable lines and capture the variables
    i = 0
    while i < number_of_vars:
        line = file.readline()
        variables.append(float(line))
    # Parse the exit code line and capture it
    exit_code = [0, 0]
    line = file.readline()
    if line and ('objno' in line):
        exit_code_line = line.split()
        if (len(exit_code_line) != 3):
            raise SolverSystemError(f"ERROR READING `sol` FILE. Expected two numbers in `objno` line; received {line}.")
        exit_code = [int(exit_code_line[1]), int(exit_code_line[2])]
    else:
        raise SolverSystemError(f"ERROR READING `sol` FILE. Expected `objno`; received {line}.")
    results.extra_info.solver_message = message.strip().replace('\n', '; ')
    # Not sure if next two lines are needed
    # if isinstance(res.solver.message, str):
    #     res.solver.message = res.solver.message.replace(':', '\\x3a')
    if (exit_code[1] >= 0) and (exit_code[1] <= 99):
        results.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
        results.solution_status = SolutionStatus.optimal
    elif (exit_code[1] >= 100) and (exit_code[1] <= 199):
        exit_code_message = "Optimal solution indicated, but ERROR LIKELY!"
        results.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
        results.solution_status = SolutionStatus.optimal
        if results.extra_info.solver_message:
            results.extra_info.solver_message += '; ' + exit_code_message
        else:
            results.extra_info.solver_message = exit_code_message
    elif (exit_code[1] >= 200) and (exit_code[1] <= 299):
        results.termination_condition = TerminationCondition.locallyInfeasible
        results.solution_status = SolutionStatus.infeasible
    elif (exit_code[1] >= 300) and (exit_code[1] <= 399):
        results.termination_condition = TerminationCondition.unbounded
        results.solution_status = SolutionStatus.infeasible
    elif (exit_code[1] >= 400) and (exit_code[1] <= 499):
        results.solver.termination_condition = TerminationCondition.iterationLimit
    elif (exit_code[1] >= 500) and (exit_code[1] <= 599):
        exit_code_message = (
            "FAILURE: the solver stopped by an error condition "
            "in the solver routines!"
        )
        results.solver.termination_condition = TerminationCondition.error
        return results
    
    return results

def parse_yaml():
    pass


def parse_json():
    pass


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
