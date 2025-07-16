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


from typing import Tuple, Dict, Any, List, Sequence, Optional, Mapping, NoReturn
import io

from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.expr import value
from pyomo.common.collections import ComponentMap
from pyomo.core.staleflag import StaleFlagManager
from pyomo.common.errors import DeveloperError, PyomoException
from pyomo.repn.plugins.nl_writer import NLWriterInfo
from pyomo.core.expr.visitor import replace_expressions
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase


class SolFileData:
    """
    Defines the data types found within a .sol file
    """

    def __init__(self) -> None:
        self.primals: List[float] = []
        self.duals: List[float] = []
        self.var_suffixes: Dict[str, Dict[int, Any]] = {}
        self.con_suffixes: Dict[str, Dict[Any]] = {}
        self.obj_suffixes: Dict[str, Dict[int, Any]] = {}
        self.problem_suffixes: Dict[str, List[Any]] = {}
        self.other: List(str) = []


class SolSolutionLoader(SolutionLoaderBase):
    """
    Loader for solvers that create .sol files (e.g., ipopt)
    """

    def __init__(self, sol_data: SolFileData, nl_info: NLWriterInfo) -> None:
        self._sol_data = sol_data
        self._nl_info = nl_info

    def load_vars(self, vars_to_load: Optional[Sequence[VarData]] = None) -> NoReturn:
        if self._nl_info is None:
            raise RuntimeError(
                'Solution loader does not currently have a valid solution. Please '
                'check results.termination_condition and/or results.solution_status.'
            )
        if self._sol_data is None:
            assert len(self._nl_info.variables) == 0
        else:
            if self._nl_info.scaling:
                for var, val, scale in zip(
                    self._nl_info.variables,
                    self._sol_data.primals,
                    self._nl_info.scaling.variables,
                ):
                    var.set_value(val / scale, skip_validation=True)
            else:
                for var, val in zip(self._nl_info.variables, self._sol_data.primals):
                    var.set_value(val, skip_validation=True)

        for var, v_expr in self._nl_info.eliminated_vars:
            var.value = value(v_expr)

        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        if self._nl_info is None:
            raise RuntimeError(
                'Solution loader does not currently have a valid solution. Please '
                'check results.termination_condition and/or results.solution_status.'
            )
        val_map = {}
        if self._sol_data is None:
            assert len(self._nl_info.variables) == 0
        else:
            if self._nl_info.scaling is None:
                scale_list = [1] * len(self._nl_info.variables)
            else:
                scale_list = self._nl_info.scaling.variables
            for var, val, scale in zip(
                self._nl_info.variables, self._sol_data.primals, scale_list
            ):
                val_map[id(var)] = val / scale

        for var, v_expr in self._nl_info.eliminated_vars:
            val = replace_expressions(v_expr, substitution_map=val_map)
            v_id = id(var)
            val_map[v_id] = val

        res = ComponentMap()
        if vars_to_load is None:
            vars_to_load = self._nl_info.variables + [
                var for var, _ in self._nl_info.eliminated_vars
            ]
        for var in vars_to_load:
            res[var] = val_map[id(var)]

        return res

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> Dict[ConstraintData, float]:
        if self._nl_info is None:
            raise RuntimeError(
                'Solution loader does not currently have a valid solution. Please '
                'check results.termination_condition and/or results.solution_status.'
            )
        if len(self._nl_info.eliminated_vars) > 0:
            raise NotImplementedError(
                'For now, turn presolve off (opt.config.writer_config.linear_presolve=False) '
                'to get dual variable values.'
            )
        if self._sol_data is None:
            raise DeveloperError(
                "Solution data is empty. This should not "
                "have happened. Report this error to the Pyomo Developers."
            )
        res = {}
        if self._nl_info.scaling is None:
            scale_list = [1] * len(self._nl_info.constraints)
            obj_scale = 1
        else:
            scale_list = self._nl_info.scaling.constraints
            obj_scale = self._nl_info.scaling.objectives[0]
        if cons_to_load is None:
            cons_to_load = set(self._nl_info.constraints)
        else:
            cons_to_load = set(cons_to_load)
        for con, val, scale in zip(
            self._nl_info.constraints, self._sol_data.duals, scale_list
        ):
            if con in cons_to_load:
                res[con] = val * scale / obj_scale
        return res


def parse_sol_file(
    sol_file: io.TextIOBase, nl_info: NLWriterInfo, result: Results
) -> Tuple[Results, SolFileData]:
    """
    Parse a .sol file and populate to Pyomo objects
    """
    sol_data = SolFileData()

    #
    # Some solvers (minto) do not write a message.  We will assume
    # all non-blank lines up to the 'Options' line is the message.
    # For backwards compatibility and general safety, we will parse all
    # lines until "Options" appears. Anything before "Options" we will
    # consider to be the solver message.
    options_found = False
    message = []
    model_objects = []
    for line in sol_file:
        if not line:
            break
        line = line.strip()
        if "Options" in line:
            # Once "Options" appears, we must now read the content under it.
            options_found = True
            line = sol_file.readline()
            number_of_options = int(line)
            # We are adding in this DeveloperError to see if the alternative case
            # is ever actually hit in the wild. In a previous iteration of the sol
            # reader, there was logic to check for the number of options, but it
            # was uncovered by tests and unclear if actually necessary.
            if number_of_options > 4:
                raise DeveloperError(
                    """
    The sol file reader has hit an unexpected error while parsing. The number of
    options recorded is greater than 4. Please report this error to the Pyomo
    developers.
    """
                )
            for i in range(number_of_options + 4):
                line = sol_file.readline()
                model_objects.append(int(line))
            break
        message.append(line)
    if not options_found:
        raise PyomoException("ERROR READING `sol` FILE. No 'Options' line found.")
    message = '\n'.join(message)
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
            raise PyomoException(
                f"ERROR READING `sol` FILE. Expected two numbers in `objno` line; received {line}."
            )
        exit_code = [int(exit_code_line[1]), int(exit_code_line[2])]
    else:
        raise PyomoException(
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
        result.solution_status = SolutionStatus.infeasible
        result.termination_condition = (
            TerminationCondition.iterationLimit
        )  # this is not always correct
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
        sol_data.primals = variable_vals
        sol_data.duals = duals
        ### Read suffixes ###
        line = sol_file.readline()
        while line:
            line = line.strip()
            if line == "":
                continue
            line = line.split()
            # Extra solver message processing
            if line[0] != 'suffix':
                # We assume this is the start of a
                # section like kestrel_option, which
                # comes after all suffixes.
                remaining = ''
                line = sol_file.readline()
                while line:
                    remaining += line.strip() + '; '
                    line = sol_file.readline()
                result.extra_info.solver_message += remaining
                break
            read_data_type = int(line[1])
            data_type = read_data_type & 3  # 0-var, 1-con, 2-obj, 3-prob
            convert_function = int
            if (read_data_type & 4) == 4:
                convert_function = float
            number_of_entries = int(line[2])
            # The third entry is name length, and it is length+1. This is unnecessary
            # except for data validation.
            # The fourth entry is table "length", e.g., memory size.
            number_of_string_lines = int(line[5])
            suffix_name = sol_file.readline().strip()
            # Add any arbitrary string lines to the "other" list
            for line in range(number_of_string_lines):
                sol_data.other.append(sol_file.readline())
            if data_type == 0:  # Var
                sol_data.var_suffixes[suffix_name] = {}
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    var_ndx = int(suf_line[0])
                    sol_data.var_suffixes[suffix_name][var_ndx] = convert_function(
                        suf_line[1]
                    )
            elif data_type == 1:  # Con
                sol_data.con_suffixes[suffix_name] = {}
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    con_ndx = int(suf_line[0])
                    sol_data.con_suffixes[suffix_name][con_ndx] = convert_function(
                        suf_line[1]
                    )
            elif data_type == 2:  # Obj
                sol_data.obj_suffixes[suffix_name] = {}
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    obj_ndx = int(suf_line[0])
                    sol_data.obj_suffixes[suffix_name][obj_ndx] = convert_function(
                        suf_line[1]
                    )
            elif data_type == 3:  # Prob
                sol_data.problem_suffixes[suffix_name] = []
                for cnt in range(number_of_entries):
                    suf_line = sol_file.readline().split()
                    sol_data.problem_suffixes[suffix_name].append(
                        convert_function(suf_line[1])
                    )
            line = sol_file.readline()

    return result, sol_data
