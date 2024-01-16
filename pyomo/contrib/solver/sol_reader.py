from typing import Tuple, Dict, Any, List
import io

from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _ConstraintData
from pyomo.core.base.objective import _ObjectiveData
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from .results import Results, SolverResultsError, SolutionStatus, TerminationCondition


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
    result: Results,
) -> Tuple[Results, SolFileData]:
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
            line = sol_file.readline()

        return result, sol_data
