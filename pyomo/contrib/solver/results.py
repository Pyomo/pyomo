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
from typing import Optional, Tuple, Dict, Any, Sequence, List
from datetime import datetime
import io

from pyomo.common.config import (
    ConfigDict,
    ConfigValue,
    Datetime,
    NonNegativeInt,
    In,
    NonNegativeFloat,
)
from pyomo.common.errors import PyomoException
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.objective import _ObjectiveData
from pyomo.opt.results.solution import SolutionStatus as LegacySolutionStatus
from pyomo.opt.results.solver import (
    TerminationCondition as LegacyTerminationCondition,
    SolverStatus as LegacySolverStatus,
)
from pyomo.contrib.solver.solution import SolutionLoaderBase
from pyomo.repn.plugins.nl_writer import NLWriterInfo


class SolverResultsError(PyomoException):
    """
    General exception to catch solver system errors
    """


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
            'incumbent_objective', ConfigValue(domain=float, default=None)
        )
        self.objective_bound: Optional[float] = self.declare(
            'objective_bound', ConfigValue(domain=float, default=None)
        )
        self.solver_name: Optional[str] = self.declare(
            'solver_name', ConfigValue(domain=str)
        )
        self.solver_version: Optional[Tuple[int, ...]] = self.declare(
            'solver_version', ConfigValue(domain=tuple)
        )
        self.iteration_count: Optional[int] = self.declare(
            'iteration_count', ConfigValue(domain=NonNegativeInt, default=None)
        )
        self.timing_info: ConfigDict = self.declare('timing_info', ConfigDict())

        self.timing_info.start_timestamp: datetime = self.timing_info.declare(
            'start_timestamp', ConfigValue(domain=Datetime)
        )
        self.timing_info.wall_time: Optional[float] = self.timing_info.declare(
            'wall_time', ConfigValue(domain=NonNegativeFloat)
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

    def report_timing(self):
        print('Timing Information: ')
        print('-' * 50)
        self.timing_info.display()
        print('-' * 50)


class ResultsReader:
    pass


class SolFileData:
    def __init__(self) -> None:
        self.primals: Dict[int, Tuple[_GeneralVarData, float]] = dict()
        self.duals: Dict[_ConstraintData, float] = dict()
        self.var_suffixes: Dict[str, Dict[int, Tuple[_GeneralVarData, Any]]] = dict()
        self.con_suffixes: Dict[str, Dict[_ConstraintData, Any]] = dict()
        self.obj_suffixes: Dict[str, Dict[int, Tuple[_ObjectiveData, Any]]] = dict()
        self.problem_suffixes: Dict[str, List[Any]] = dict()


def parse_sol_file(
    sol_file: io.TextIOBase,
    nl_info: NLWriterInfo,
    suffixes_to_read: Sequence[str],
    result: Results,
) -> Tuple[Results, SolFileData]:
    suffixes_to_read = set(suffixes_to_read)
    sol_data = SolFileData()

    #
    # Some solvers (minto) do not write a message.  We will assume
    # all non-blank lines up the 'Options' line is the message.
    # For backwards compatibility and general safety, we will parse all
    # lines until "Options" appears. Anything before "Options" we will
    # consider to be the solver message.
    message = []
    for line in sol_file:
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
        line = sol_file.readline()
        number_of_options = int(line)
        need_tolerance = False
        if (
            number_of_options > 4
        ):  # MRM: Entirely unclear why this is necessary, or if it even is
            number_of_options -= 2
            need_tolerance = True
        for i in range(number_of_options + 4):
            line = sol_file.readline()
            model_objects.append(int(line))
        if (
            need_tolerance
        ):  # MRM: Entirely unclear why this is necessary, or if it even is
            line = sol_file.readline()
            model_objects.append(float(line))
    else:
        raise SolverResultsError("ERROR READING `sol` FILE. No 'Options' line found.")
    # Identify the total number of variables and constraints
    number_of_cons = model_objects[number_of_options + 1]
    number_of_vars = model_objects[number_of_options + 3]
    assert number_of_cons == len(nl_info.constraints)
    assert number_of_vars == len(nl_info.variables)

    duals = [float(sol_file.readline()) for i in range(number_of_cons)]
    variable_vals = [float(sol_file.readline()) for i in range(number_of_vars)]

    # Parse the exit code line and capture it
    exit_code = [0, 0]
    line = sol_file.readline()
    if line and ('objno' in line):
        exit_code_line = line.split()
        if len(exit_code_line) != 3:
            raise SolverResultsError(
                f"ERROR READING `sol` FILE. Expected two numbers in `objno` line; received {line}."
            )
        exit_code = [int(exit_code_line[1]), int(exit_code_line[2])]
    else:
        raise SolverResultsError(
            f"ERROR READING `sol` FILE. Expected `objno`; received {line}."
        )
    result.extra_info.solver_message = message.strip().replace('\n', '; ')
    exit_code_message = ''
    if (exit_code[1] >= 0) and (exit_code[1] <= 99):
        result.solution_status = SolutionStatus.optimal
        result.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
    elif (exit_code[1] >= 100) and (exit_code[1] <= 199):
        exit_code_message = "Optimal solution indicated, but ERROR LIKELY!"
        result.solution_status = SolutionStatus.feasible
        result.termination_condition = TerminationCondition.error
    elif (exit_code[1] >= 200) and (exit_code[1] <= 299):
        exit_code_message = "INFEASIBLE SOLUTION: constraints cannot be satisfied!"
        result.solution_status = SolutionStatus.infeasible
        # TODO: this is solver dependent
        # But this was the way in the previous version - and has been fine thus far?
        result.termination_condition = TerminationCondition.locallyInfeasible
    elif (exit_code[1] >= 300) and (exit_code[1] <= 399):
        exit_code_message = (
            "UNBOUNDED PROBLEM: the objective can be improved without limit!"
        )
        result.solution_status = SolutionStatus.noSolution
        result.termination_condition = TerminationCondition.unbounded
    elif (exit_code[1] >= 400) and (exit_code[1] <= 499):
        exit_code_message = (
            "EXCEEDED MAXIMUM NUMBER OF ITERATIONS: the solver "
            "was stopped by a limit that you set!"
        )
        # TODO: this is solver dependent
        # But this was the way in the previous version - and has been fine thus far?
        result.solution_status = SolutionStatus.infeasible
        result.termination_condition = TerminationCondition.iterationLimit
    elif (exit_code[1] >= 500) and (exit_code[1] <= 599):
        exit_code_message = (
            "FAILURE: the solver stopped by an error condition "
            "in the solver routines!"
        )
        result.termination_condition = TerminationCondition.error

    if result.extra_info.solver_message:
        if exit_code_message:
            result.extra_info.solver_message += '; ' + exit_code_message
    else:
        result.extra_info.solver_message = exit_code_message

    if result.solution_status != SolutionStatus.noSolution:
        for v, val in zip(nl_info.variables, variable_vals):
            sol_data.primals[id(v)] = (v, val)
        if "dual" in suffixes_to_read:
            for c, val in zip(nl_info.constraints, duals):
                sol_data.duals[c] = val
        ### Read suffixes ###
        line = sol_file.readline()
        while line:
            line = line.strip()
            if line == "":
                continue
            line = line.split()
            # Some sort of garbage we tag onto the solver message, assuming we are past the suffixes
            if line[0] != 'suffix':
                # We assume this is the start of a
                # section like kestrel_option, which
                # comes after all suffixes.
                remaining = ""
                line = sol_file.readline()
                while line:
                    remaining += line.strip() + "; "
                    line = sol_file.readline()
                result.extra_info.solver_message += remaining
                break
            unmasked_kind = int(line[1])
            kind = unmasked_kind & 3  # 0-var, 1-con, 2-obj, 3-prob
            convert_function = int
            if (unmasked_kind & 4) == 4:
                convert_function = float
            nvalues = int(line[2])
            # namelen = int(line[3])
            # tablen = int(line[4])
            tabline = int(line[5])
            suffix_name = sol_file.readline().strip()
            if suffix_name in suffixes_to_read:
                # ignore translation of the table number to string value for now,
                # this information can be obtained from the solver documentation
                for n in range(tabline):
                    sol_file.readline()
                if kind == 0:  # Var
                    sol_data.var_suffixes[suffix_name] = dict()
                    for cnt in range(nvalues):
                        suf_line = sol_file.readline().split()
                        var_ndx = int(suf_line[0])
                        var = nl_info.variables[var_ndx]
                        sol_data.var_suffixes[suffix_name][id(var)] = (
                            var,
                            convert_function(suf_line[1]),
                        )
                elif kind == 1:  # Con
                    sol_data.con_suffixes[suffix_name] = dict()
                    for cnt in range(nvalues):
                        suf_line = sol_file.readline().split()
                        con_ndx = int(suf_line[0])
                        con = nl_info.constraints[con_ndx]
                        sol_data.con_suffixes[suffix_name][con] = convert_function(
                            suf_line[1]
                        )
                elif kind == 2:  # Obj
                    sol_data.obj_suffixes[suffix_name] = dict()
                    for cnt in range(nvalues):
                        suf_line = sol_file.readline().split()
                        obj_ndx = int(suf_line[0])
                        obj = nl_info.objectives[obj_ndx]
                        sol_data.obj_suffixes[suffix_name][id(obj)] = (
                            obj,
                            convert_function(suf_line[1]),
                        )
                elif kind == 3:  # Prob
                    sol_data.problem_suffixes[suffix_name] = list()
                    for cnt in range(nvalues):
                        suf_line = sol_file.readline().split()
                        sol_data.problem_suffixes[suffix_name].append(
                            convert_function(suf_line[1])
                        )
            else:
                # do not store the suffix in the solution object
                for cnt in range(nvalues):
                    sol_file.readline()
            line = sol_file.readline()

        return result, sol_data


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
