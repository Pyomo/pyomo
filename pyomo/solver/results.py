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
import re
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
from pyomo.common.collections import ComponentMap
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.objective import _ObjectiveData
from pyomo.opt.results.solution import SolutionStatus as LegacySolutionStatus
from pyomo.opt.results.solver import (
    TerminationCondition as LegacyTerminationCondition,
    SolverStatus as LegacySolverStatus,
)
from pyomo.solver.solution import SolutionLoaderBase
from pyomo.solver.util import SolverSystemError
from pyomo.repn.plugins.nl_writer import NLWriterInfo


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

        self.timing_info.start_time: datetime = self.timing_info.declare(
            'start_time', ConfigValue(domain=Datetime)
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
        self.solver_message: Optional[str] = self.declare(
            'solver_message',
            ConfigValue(domain=str, default=None),
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


class SolFileData(object):
    def __init__(self) -> None:
        self.primals: Dict[int, Tuple[_GeneralVarData, float]] = dict()
        self.duals: Dict[_ConstraintData, float] = dict()
        self.var_suffixes: Dict[str, Dict[int, Tuple[_GeneralVarData, Any]]] = dict()
        self.con_suffixes: Dict[str, Dict[_ConstraintData, Any]] = dict()
        self.obj_suffixes: Dict[str, Dict[int, Tuple[_ObjectiveData, Any]]] = dict()
        self.problem_suffixes: Dict[str, List[Any]] = dict()


def parse_sol_file(sol_file: io.TextIOBase, nl_info: NLWriterInfo, suffixes_to_read: Sequence[str]) -> Tuple[Results, SolFileData]:
    suffixes_to_read = set(suffixes_to_read)
    res = Results()
    sol_data = SolFileData()

    fin = sol_file
    #
    # Some solvers (minto) do not write a message.  We will assume
    # all non-blank lines up the 'Options' line is the message.
    msg = []
    while True:
        line = fin.readline()
        if not line:
            # EOF
            break
        line = line.strip()
        if line == 'Options':
            break
        if line:
            msg.append(line)
    msg = '\n'.join(msg)
    z = []
    if line[:7] == "Options":
        line = fin.readline()
        nopts = int(line)
        need_vbtol = False
        if nopts > 4:  # WEH - when is this true?
            nopts -= 2
            need_vbtol = True
        for i in range(nopts + 4):
            line = fin.readline()
            z += [int(line)]
        if need_vbtol:  # WEH - when is this true?
            line = fin.readline()
            z += [float(line)]
    else:
        raise ValueError("no Options line found")
    n = z[nopts + 3]  # variables
    m = z[nopts + 1]  # constraints
    x = []
    y = []
    i = 0
    while i < m:
        line = fin.readline()
        y.append(float(line))
        i += 1
    i = 0
    while i < n:
        line = fin.readline()
        x.append(float(line))
        i += 1
    objno = [0, 0]
    line = fin.readline()
    if line:  # WEH - when is this true?
        if line[:5] != "objno":  # pragma:nocover
            raise ValueError("expected 'objno', found '%s'" % (line))
        t = line.split()
        if len(t) != 3:
            raise ValueError(
                "expected two numbers in objno line, but found '%s'" % (line)
            )
        objno = [int(t[1]), int(t[2])]
    res.solver_message = msg.strip().replace("\n", "; ")
    res.solution_status = SolutionStatus.noSolution
    res.termination_condition = TerminationCondition.unknown
    if (objno[1] >= 0) and (objno[1] <= 99):
        res.solution_status = SolutionStatus.optimal
        res.termination_condition = TerminationCondition.convergenceCriteriaSatisfied
    elif (objno[1] >= 100) and (objno[1] <= 199):
        res.solution_status = SolutionStatus.feasible
        res.termination_condition = TerminationCondition.error
    elif (objno[1] >= 200) and (objno[1] <= 299):
        res.solution_status = SolutionStatus.infeasible
        # TODO: this is solver dependent
        res.termination_condition = TerminationCondition.locallyInfeasible
    elif (objno[1] >= 300) and (objno[1] <= 399):
        res.solution_status = SolutionStatus.noSolution
        res.termination_condition = TerminationCondition.unbounded
    elif (objno[1] >= 400) and (objno[1] <= 499):
        # TODO: this is solver dependent
        res.solution_status = SolutionStatus.infeasible
        res.termination_condition = TerminationCondition.iterationLimit
    elif (objno[1] >= 500) and (objno[1] <= 599):
        res.solution_status = SolutionStatus.noSolution
        res.termination_condition = TerminationCondition.error
    if res.solution_status != SolutionStatus.noSolution:
        for v, val in zip(nl_info.variables, x):
            sol_data[id(v)] = (v, val)
        if "dual" in suffixes_to_read:
            for c, val in zip(nl_info.constraints, y):
                sol_data[c] = val
        ### Read suffixes ###
        line = fin.readline()
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
                line = fin.readline()
                while line:
                    remaining += line.strip() + "; "
                    line = fin.readline()
                res.solver_message += remaining
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
            suffix_name = fin.readline().strip()
            if suffix_name in suffixes_to_read:
                # ignore translation of the table number to string value for now,
                # this information can be obtained from the solver documentation
                for n in range(tabline):
                    fin.readline()
                if kind == 0:  # Var
                    sol_data.var_suffixes[suffix_name] = dict()
                    for cnt in range(nvalues):
                        suf_line = fin.readline().split()
                        var_ndx = int(suf_line[0])
                        var = nl_info.variables[var_ndx]
                        sol_data.var_suffixes[suffix_name][id(var)] = (var, convert_function(suf_line[1]))
                elif kind == 1:  # Con
                    sol_data.con_suffixes[suffix_name] = dict()
                    for cnt in range(nvalues):
                        suf_line = fin.readline().split()
                        con_ndx = int(suf_line[0])
                        con = nl_info.constraints[con_ndx]
                        sol_data.con_suffixes[suffix_name][con] = convert_function(suf_line[1])
                elif kind == 2:  # Obj
                    sol_data.obj_suffixes[suffix_name] = dict()
                    for cnt in range(nvalues):
                        suf_line = fin.readline().split()
                        obj_ndx = int(suf_line[0])
                        obj = nl_info.objectives[obj_ndx]
                        sol_data.obj_suffixes[suffix_name][id(obj)] = (obj, convert_function(suf_line[1]))
                elif kind == 3:  # Prob
                    sol_data.problem_suffixes[suffix_name] = list()
                    for cnt in range(nvalues):
                        suf_line = fin.readline().split()
                        sol_data.problem_suffixes[suffix_name].append(convert_function(suf_line[1]))
            else:
                # do not store the suffix in the solution object
                for cnt in range(nvalues):
                    fin.readline()
            line = fin.readline()

        return res, sol_data


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
