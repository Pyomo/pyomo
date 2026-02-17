# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import io
from typing import Sequence, Optional, Mapping

from pyomo.common.collections import ComponentMap
from pyomo.common.errors import MouseTrap
from pyomo.core.base.constraint import ConstraintData
from pyomo.core.base.var import VarData
from pyomo.core.expr import value
from pyomo.core.staleflag import StaleFlagManager
from pyomo.core.expr.visitor import replace_expressions
from pyomo.repn.plugins.nl_writer import NLWriterInfo

from pyomo.contrib.solver.common.util import SolverError
from pyomo.contrib.solver.common.results import (
    Results,
    SolutionStatus,
    TerminationCondition,
)
from pyomo.contrib.solver.common.solution_loader import SolutionLoaderBase


class ASLSolFileData:
    """
    Defines the data fields found within an ASL .sol file
    """

    def __init__(self) -> None:
        self.message: str = None
        self.objno: int = 0
        self.solve_code: int = None
        self.ampl_options: list[int | float] = None
        self.primals: list[float] = None
        self.duals: list[float] = None
        self.var_suffixes: dict[str, dict[int, int | float]] = {}
        self.con_suffixes: dict[str, dict[int, int | float]] = {}
        self.obj_suffixes: dict[str, dict[int, int | float]] = {}
        self.problem_suffixes: dict[str, int | float] = {}
        self.suffix_table: dict[(int, str), list[int | float, str, ...]] = {}
        self.unparsed: str = None


class ASLSolFileSolutionLoader(SolutionLoaderBase):
    """
    Loader for solvers that create ASL .sol files (e.g., ipopt)
    """

    def __init__(self, sol_data: ASLSolFileData, nl_info: NLWriterInfo) -> None:
        self._sol_data = sol_data
        self._nl_info = nl_info

    def load_vars(self, vars_to_load: Optional[Sequence[VarData]] = None) -> None:
        if vars_to_load is not None:
            # If we are given a list of variables to load, it is easiest
            # to use the filtering in get_primals and then just set
            # those values.
            for var, val in self.get_primals(vars_to_load).items():
                var.set_value(val, skip_validation=True)
            StaleFlagManager.mark_all_as_stale(delayed=True)
            return

        if not self._sol_data.primals:
            # SOL file contained no primal values
            assert len(self._nl_info.variables) == 0
        else:
            # Load the primals provided by the SOL file (scaling if necessary)
            assert len(self._nl_info.variables) == len(self._sol_data.primals)
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

        # Compute all variables presolved out of the model
        for var, v_expr in self._nl_info.eliminated_vars:
            var.set_value(value(v_expr), skip_validation=True)

        StaleFlagManager.mark_all_as_stale(delayed=True)

    def get_primals(
        self, vars_to_load: Optional[Sequence[VarData]] = None
    ) -> Mapping[VarData, float]:
        result = ComponentMap()
        if not self._sol_data.primals:
            # SOL file contained no primal values
            assert len(self._nl_info.variables) == 0
        else:
            # Load the primals provided by the SOL file (scaling if necessary)
            assert len(self._nl_info.variables) == len(self._sol_data.primals)
            if self._nl_info.scaling:
                for var, val, scale in zip(
                    self._nl_info.variables,
                    self._sol_data.primals,
                    self._nl_info.scaling.variables,
                ):
                    result[var] = val / scale
            else:
                for var, val in zip(self._nl_info.variables, self._sol_data.primals):
                    result[var] = val

        # If we have eliminated variables, then we need to compute
        # them.  Unfortunately, the expressions that we kept are in
        # terms of the actual variable values (which we don't want to
        # modify).  We will make use of an expression replacement
        # visitor to perform the substitution and computation.
        #
        # It would be great if we could do this without creating the
        # entire (unfiltered) result, but we just don't (easily) know
        # which variable values we are going to need (either in the
        # vars_to_load list, or in any expression that might be needed
        # to compute an eliminated variable value.  So to keep things
        # simple (i.e., fewer bugs), we will go ahead and always compute
        # everything.
        if self._nl_info.eliminated_vars:
            val_map = {id(k): v for k, v in result.items()}
            for var, v_expr in self._nl_info.eliminated_vars:
                val = value(replace_expressions(v_expr, substitution_map=val_map))
                val_map[id(var)] = val
                result[var] = val

        if vars_to_load is not None:
            result = ComponentMap((v, result[v]) for v in vars_to_load)

        return result

    def get_duals(
        self, cons_to_load: Optional[Sequence[ConstraintData]] = None
    ) -> dict[ConstraintData, float]:
        if len(self._nl_info.eliminated_vars) > 0:
            raise MouseTrap(
                'Complete duals are not available when variables have '
                'been presolved from the model.  Turn presolve off '
                '(solver.config.writer_config.linear_presolve=False) to get '
                'dual variable values.'
            )

        scaling = self._nl_info.scaling
        if scaling:
            _iter = zip(
                self._nl_info.constraints, self._sol_data.duals, scaling.constraints
            )
            inv_obj_scale = 1.0
            if self._nl_info.scaling.objectives:
                inv_obj_scale /= self._nl_info.scaling.objectives[self._sol_data.objno]
        else:
            _iter = zip(self._nl_info.constraints, self._sol_data.duals)
        if cons_to_load is not None:
            cons_to_load = set(cons_to_load)
            _iter = filter(lambda x: x[0] in cons_to_load, _iter)
        if scaling:
            return {con: val * scale * inv_obj_scale for con, val, scale in _iter}
        else:
            return {con: val for con, val in _iter}


def asl_solve_code_to_solution_status(
    sol_data: ASLSolFileData, result: Results
) -> None:
    """Convert the ASL "solve code" integer into a Pyomo status

    The ASL returns an indication of the solution status and termination
    condition using a single "solve code" integer.  This function
    implements the translation of the numeric value into the Pyomo
    equivalents (:class:`TerminationCondition` and
    :class:`SolutionStatus`), as well as a general string description,
    using the table from Section 14.2 in the AMPL Book [FGK02]_.

    """
    #
    # This table (the values and the string interpretations) are from
    # Chapter 14 in the AMPL Book:
    #
    code = sol_data.solve_code
    status = SolutionStatus.unknown if sol_data.primals else SolutionStatus.noSolution
    if code is None:
        message = f"AMPL({code}): solver did not generate a SOL file"
        term = TerminationCondition.error
    elif (code >= 0) and (code <= 99):
        # message = f"AMPL({code}:solved): optimal solution found"
        message = ''
        status = SolutionStatus.optimal
        term = TerminationCondition.convergenceCriteriaSatisfied
    elif (code >= 100) and (code <= 199):
        message = f"AMPL({code}:solved?): optimal solution indicated, but error likely"
        status = SolutionStatus.feasible
        term = TerminationCondition.error
    elif (code >= 200) and (code <= 299):
        message = f"AMPL({code}:infeasible): constraints cannot be satisfied"
        status = SolutionStatus.infeasible
        term = TerminationCondition.locallyInfeasible
    elif (code >= 300) and (code <= 399):
        message = f"AMPL({code}:unbounded): objective can be improved without limit"
        term = TerminationCondition.unbounded
    elif (code >= 400) and (code <= 499):
        message = f"AMPL({code}:limit): stopped by a limit that you set"
        term = TerminationCondition.iterationLimit  # this is not always correct
    elif (code >= 500) and (code <= 599):
        message = f"AMPL({code}:failure): stopped by an error condition in the solver"
        term = TerminationCondition.error
    else:
        message = f"AMPL({code}): unexpected solve code"
        term = TerminationCondition.error

    if sol_data.message:
        # TBD: [JDS 10/2025]: Why do we convert newlines to semicolons?
        result.extra_info.solver_message = sol_data.message.replace('\n', '; ')
        if message:
            result.extra_info.solver_message += '; ' + message
    else:
        result.extra_info.solver_message = message
    result.solution_status = status
    result.termination_condition = term


def parse_asl_sol_file(FILE: io.TextIOBase) -> ASLSolFileData:
    """Parse an ASL .sol file.

    This is a standalone routine to parse the AMPL Solver Library (ASL)
    "``.sol``" file format.  The resulting :class:`ASLSolFileData`
    object is a faithful representation of the data from the file.
    Translating the parsed data back into the context of a Pyomo model
    requires additional information (at the very least, the Pyomo model
    and the :class:`NLWriterInfo` data structure generated by the writer
    that originally created the ``.nl`` file that was sent to the
    solver.

    """
    sol_data = ASLSolFileData()

    # Parse the initial solver message and the AMPL options sections
    z = _parse_message_and_options(FILE, sol_data)

    #
    # Parse the duals and variable values
    #
    num_duals = z[1]  # "m" in writesol.c
    assert num_duals == z[0] or not num_duals
    sol_data.duals = [float(FILE.readline()) for i in range(num_duals)]

    num_primals = z[3]  # "n" in writesol.c
    assert num_primals == z[2] or not num_primals
    sol_data.primals = [float(FILE.readline()) for i in range(num_primals)]

    # Parse the OBJNO (objective number and solver exit code)
    _parse_objno_and_exitcode(FILE, sol_data)

    # Parse the suffix data
    _parse_suffixes(FILE, sol_data)

    return sol_data


def _parse_message_and_options(FILE: io.TextIOBase, data: ASLSolFileData) -> list[int]:
    msg = []
    # Some solvers (minto) do not write a message.  We will assume
    # all non-blank lines up the 'Options' line is the message.
    while True:
        line = FILE.readline()
        if not line:
            # EOF
            raise SolverError("Error reading `sol` file: no 'Options' line found.")
        line = line.strip()
        if line == 'Options':
            break
        if line:
            msg.append(line)
    data.message = "\n".join(msg)

    # WARNING: This appears to be undocumented outside of the ASL
    # writesol.c implementation.  Before changing this logic, please
    # familiarize yourself with that code.
    #
    # The AMPL options are a sequence of ints, the first of which
    # specifies the number of options to expect, followed by the
    # options (all ints), followed by the 4 int-elements of "z".
    #
    n_opts = int(FILE.readline())
    #
    # The ASL will occasionally "lie" about the number of options: if
    # the second option (not including the number of options) is "3",
    # then the ASL will add 2 to the number of options reported, and
    # will add *one* option (vbtol, a float) *after* the elements of
    # "z".
    #
    # Because of this, we will read the first two options from the file
    # first so we can know how to correctly parse the remaining options.
    assert n_opts >= 2
    ampl_options = [int(FILE.readline()), int(FILE.readline())]
    read_vbtol = ampl_options[1] == 3
    if read_vbtol:
        n_opts -= 2
    ampl_options.extend(int(FILE.readline()) for i in range(n_opts - 2))
    # Note: "z" comes from the name used for this data structure in
    # `writesol.c`.  It is unknown to us what motivated that name.
    #
    # Z: [ #cons; #duals, #vars, #var_vals ]
    #    #duals will either be #cons or 0
    #    #var_vals will either be #vars or 0
    z = [int(FILE.readline()) for i in range(4)]
    if read_vbtol:
        ampl_options.append(float(FILE.readline()))

    data.ampl_options = ampl_options
    return z


def _parse_objno_and_exitcode(FILE: io.TextIOBase, data: ASLSolFileData) -> None:
    line = FILE.readline().strip()
    objno = line.split(maxsplit=2)
    if not objno or objno[0] != 'objno':
        raise SolverError(
            f"Error reading `sol` file: expected 'objno'; received {line!r}."
        )
    elif len(objno) != 3:
        # TBD: [JDS, 10/2025] there are paths where writesol.c will
        # generate `objno` lines that contain only the objective number
        # and not the solve_code.  It is not clear to me that we should
        # generate an exception here.
        raise SolverError(
            "Error reading `sol` file: expected two numbers in 'objno' line; "
            f"received {line!r}."
        )
    data.objno = int(objno[1])
    data.solve_code = int(objno[2])


def _parse_suffixes(FILE: io.TextIOBase, data: ASLSolFileData) -> None:
    while line := FILE.readline():
        line = line.strip()
        if not line:
            continue

        line = line.split(maxsplit=6)
        if line[0] != 'suffix':
            # We assume this is the start of a section (like
            # kestrel_option) that comes *after* all suffixes.  We
            # will capture it (and everything after it) and return
            # it as a single "unparsed" text string.
            data.unparsed = ' '.join(line) + "\n" + ''.join(FILE)
            break

        # Each suffix is introduced by:
        #
        #   'suffix' <kind> <n> <namelen> <tablelen> <tablines>
        #   <sufname>
        #
        # Where:
        #   kind (int): bitmask indicating suffix data type and target
        #   n (int): number of values returned
        #   namelen (int): suffix name string length (including NULL termination)
        #   tablelen (int): length of the "table" string (including NULL)
        #   tablines (int): number of lines in the table
        #   sufname (str): suffix name
        kind = int(line[1])
        value_converter = float if kind & 4 else int
        suffix_target = kind & 3  # 0-var, 1-con, 2-obj, 3-prob

        num_values = int(line[2])
        # Note: we will use namelen to strip off the newline instead of
        # strip() in case the suffix name actually ended with whitespace
        # (evil, but technically allowed by the NL spec)
        suffix_name = FILE.readline()[: int(line[3]) - 1]

        # If the Suffix includes a value <-> string table, parse it.
        # The table should be a series of <tablines> lines of the form:
        #
        #     <val> <str> <description>
        #
        # The string representation of a suffix value is the row in the
        # table whose <val> is the largest value less than or equal to
        # the suffix value.  The table should be ordered by <val>.
        if int(line[4]):
            data.suffix_table[suffix_target, suffix_name] = [
                FILE.readline().strip().split(maxsplit=2) for _ in range(int(line[5]))
            ]
            for entry in data.suffix_table[suffix_target, suffix_name]:
                entry[0] = value_converter(entry[0])

        # Parse the actual suffix values
        if suffix_target == 0:  # Var
            data.var_suffixes[suffix_name] = suffix = {}
        elif suffix_target == 1:  # Con
            data.con_suffixes[suffix_name] = suffix = {}
        elif suffix_target == 2:  # Obj
            data.obj_suffixes[suffix_name] = suffix = {}
        elif suffix_target == 3:  # Prob
            suffix = {}
        # else:  # Unreachable: kind & 3 can ONLY be 0..3

        for cnt in range(num_values):
            suf_line = FILE.readline().split(maxsplit=1)
            suffix[int(suf_line[0])] = value_converter(suf_line[1])

        if suffix_target == 3 and suffix:
            assert len(suffix) == 1
            data.problem_suffixes[suffix_name] = next(iter(suffix.values()))

    return
