#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import os

import pyutilib
import pyutilib.th as unittest

from pyomo.core import Binary, ConcreteModel, Constraint, Objective, Var, Integers, RangeSet, minimize, quicksum, Suffix
from pyomo.opt import (BranchDirection, ProblemFormat, SolverFactory,
                       SolverStatus, TerminationCondition, convert_problem)
from pyomo.solvers.plugins.solvers.CPLEX import CPLEXSHELL, MockCPLEX, _validate_file_name, ORDFileSchema


class _mock_cplex_128(object):
    def version(self):
        return (12,8,0)

class _mock_cplex_126(object):
    def version(self):
        return (12,6,0)

class CPLEX_utils(unittest.TestCase):
    def test_validate_file_name(self):
        _126 = _mock_cplex_126()
        _128 = _mock_cplex_128()

        # Check plain file
        fname = 'foo.lp'
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))

        # Check spaces in the file
        fname = 'foo bar.lp'
        with self.assertRaisesRegexp(
                ValueError, "Space detected in CPLEX xxx file"):
            _validate_file_name(_126, fname, 'xxx')
        self.assertEqual('"%s"' % (fname,),
                         _validate_file_name(_128, fname, 'xxx'))

        # check OK path separators
        fname = 'foo%sbar.lp' % (os.path.sep,)
        self.assertEqual(fname, _validate_file_name(_126, fname, 'xxx'))
        self.assertEqual(fname, _validate_file_name(_128, fname, 'xxx'))

        # check BAD path separators
        bad_char = '/\\'.replace(os.path.sep,'')
        fname = 'foo%sbar.lp' % (bad_char,)
        msg = 'Unallowed character \(%s\) found in CPLEX xxx file' % (
            repr(bad_char)[1:-1],)
        with self.assertRaisesRegexp(ValueError, msg):
            _validate_file_name(_126, fname, 'xxx')
        with self.assertRaisesRegexp(ValueError, msg):
            _validate_file_name(_128, fname, 'xxx')


class CPLEXShellWritePrioritiesFile(unittest.TestCase):
    def setUp(self):
        from pyomo.solvers.plugins.converter.model import PyomoMIPConverter  # register the `ProblemConverterFactory`
        from pyomo.repn.plugins.cpxlp import ProblemWriter_cpxlp  # register the `WriterFactory`

        self.mock_model = self.get_mock_model()
        self.mock_cplex_shell = self.get_mock_cplex_shell(self.mock_model)
        self.mock_cplex_shell._priorities_file_name = pyutilib.services.TempfileManager.create_tempfile(
            suffix=".cplex.ord"
        )

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()

    def get_mock_model(self):
        model = ConcreteModel()
        model.x = Var(within=Binary)
        model.con = Constraint(expr=model.x >= 1)
        model.obj = Objective(expr=model.x)
        return model

    def get_mock_cplex_shell(self, mock_model):
        solver = MockCPLEX()
        solver._problem_files, solver._problem_format, solver._smap_id = convert_problem(
            (mock_model,),
            ProblemFormat.cpxlp,
            [ProblemFormat.cpxlp],
            has_capability=lambda x: True,
        )
        return solver

    def get_priorities_file_as_string(self, mock_cplex_shell):
        with open(mock_cplex_shell._priorities_file_name, "r") as ord_file:
            priorities_file = ord_file.read()
        return priorities_file

    def test_write_without_priority_suffix(self):
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_write_priority_to_priorities_file(self):
        self.mock_model.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self.mock_model.priority.set_value(self.mock_model.x, priority_val)

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x1 %d\nENDATA\n"
            % (priority_val,),
        )

    def test_write_priority_and_direction_to_priorities_file(self):
        self.mock_model.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self.mock_model.priority.set_value(self.mock_model.x, priority_val)

        self.mock_model.direction = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        direction_val = BranchDirection.down
        self.mock_model.direction.set_value(self.mock_model.x, direction_val)

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\n %s x1 %d\nENDATA\n"
            % (ORDFileSchema._direction_to_str(direction_val), priority_val),
        )

    def test_raise_due_to_invalid_priority(self):
        self.mock_model.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        self.mock_model.priority.set_value(self.mock_model.x, -1)
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

        self.mock_model.priority.set_value(self.mock_model.x, 1.1)
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_use_default_due_to_invalid_direction(self):
        self.mock_model.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self.mock_model.priority.set_value(self.mock_model.x, priority_val)

        self.mock_model.direction = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        self.mock_model.direction.set_value(
            self.mock_model.x, "invalid_branching_direction"
        )

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x1 %d\nENDATA\n"
            % (priority_val,),
        )


class CPLEXShellSolvePrioritiesFile(unittest.TestCase):
    def setUp(self):
        pass

    def get_mock_model_with_priorities(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers)
        m.s = RangeSet(10)
        m.y = Var(m.s, domain=Integers)
        m.o = Objective(expr=m.x + sum(m.y), sense=minimize)
        m.c = Constraint(expr=m.x >= 1)
        m.c2 = Constraint(expr=quicksum(m.y[i] for i in m.s) >= 10)

        m.priority = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        m.direction = Suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)

        m.priority.set_value(m.x, 1)

        # Ensure tests work for both options of `expand`
        m.priority.set_value(m.y, 2, expand=False)
        m.direction.set_value(m.y, BranchDirection.down, expand=True)

        m.direction.set_value(m.y[10], BranchDirection.up)
        return m

    def test_use_variable_priorities(self):
        model = self.get_mock_model_with_priorities()
        with SolverFactory("_mock_cplex") as opt:
            opt._presolve(model, priorities=True, keepfiles=True)

            with open(opt._priorities_file_name, "r") as ord_file:
                priorities_file = ord_file.read()

        assert priorities_file == (
            "* ENCODING=ISO-8859-1\n"
            "NAME             Priority Order\n"
            "  x1 1\n"
            " DN x2 2\n"
            " DN x3 2\n"
            " DN x4 2\n"
            " DN x5 2\n"
            " DN x6 2\n"
            " DN x7 2\n"
            " DN x8 2\n"
            " DN x9 2\n"
            " DN x10 2\n"
            " UP x11 2\n"
            "ENDATA\n"
        )

        assert "read %s\n" % (opt._priorities_file_name,) in opt._command.script

    def test_ignore_variable_priorities(self):
        model = self.get_mock_model_with_priorities()
        with SolverFactory("_mock_cplex") as opt:
            opt._presolve(model, priorities=False, keepfiles=True)

            assert opt._priorities_file_name is None
            assert ".ord" not in opt._command.script

    def test_can_use_manual_priorities_file_with_lp_solve(self):
        """ Test that we can pass an LP file (not a pyomo model) along with a priorities file to `.solve()` """
        model = self.get_mock_model_with_priorities()

        with SolverFactory("_mock_cplex") as pre_opt:
            pre_opt._presolve(model, priorities=True, keepfiles=True)
            lp_file = pre_opt._problem_files[0]
            priorities_file_name = pre_opt._priorities_file_name

            with open(priorities_file_name, "r") as ord_file:
                provided_priorities_file = ord_file.read()

        with SolverFactory("_mock_cplex") as opt:
            opt._presolve(
                lp_file,
                priorities=True,
                priorities_file=priorities_file_name,
                keepfiles=True,
            )

            assert ".ord" in opt._command.script

            with open(opt._priorities_file_name, "r") as ord_file:
                priorities_file = ord_file.read()

        assert priorities_file == provided_priorities_file


class TestCPLEXSHELLProcessLogfile(unittest.TestCase):
    def setUp(self):
        solver = MockCPLEX()
        solver._log_file = pyutilib.services.TempfileManager.create_tempfile(
            suffix=".log"
        )
        self.solver = solver

    def tearDown(self):
        pyutilib.services.TempfileManager.clear_tempfiles()

    def test_log_file_shows_no_solution(self):
        log_file_text = """
MIP - Time limit exceeded, no integer solution.
Current MIP best bound =  0.0000000000e+00 (gap is infinite)
Solution time =    0.00 sec.  Iterations = 0  Nodes = 0
Deterministic time = 0.00 ticks  (0.20 ticks/sec)

CPLEX> CPLEX Error  1217: No solution exists.
No file written.
CPLEX>"""
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.noSolution
        )
        self.assertEqual(
            results.solver.termination_message,
            "MIP - Time limit exceeded, no integer solution.",
        )
        self.assertEqual(results.solver.return_code, 1217)

    def test_log_file_shows_infeasible(self):
        log_file_text = """
MIP - Integer infeasible.
Current MIP best bound =  0.0000000000e+00 (gap is infinite)
Solution time =    0.00 sec.  Iterations = 0  Nodes = 0
Deterministic time = 0.00 ticks  (0.20 ticks/sec)

CPLEX> CPLEX Error  1217: No solution exists.
No file written.
CPLEX>"""
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(
            results.solver.termination_message, "MIP - Integer infeasible."
        )
        self.assertEqual(results.solver.return_code, 1217)

    def test_log_file_shows_presolve_infeasible(self):
        log_file_text = """
Infeasibility row 'c_e_x18_':  0  = -1.
Presolve time = 0.00 sec. (0.00 ticks)
Presolve - Infeasible.
Solution time =    0.00 sec.
Deterministic time = 0.00 ticks  (0.61 ticks/sec)
CPLEX> CPLEX Error  1217: No solution exists.
No file written.
CPLEX>"""

        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.status, SolverStatus.warning)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.infeasible
        )
        self.assertEqual(
            results.solver.termination_message, "Presolve - Infeasible."
        )
        self.assertEqual(results.solver.return_code, 1217)

    def test_log_file_shows_max_time_limit_exceeded_with_feasible_solution(self):
        log_file_text = """
MIP - Time limit exceeded, integer feasible:  Objective =  0.0000000000e+00
Current MIP best bound =  0.0000000000e+00 (gap = 10.0, 10.00%)
Solution time =   10.00 sec.  Iterations = 10000  Nodes = 1000
Deterministic time = 100.00 ticks  (10.00 ticks/sec)

CPLEX> Incumbent solution written to file '/var/folders/_x/xxx/T/tmpxxx.cplex.sol'.
CPLEX>"""
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxTimeLimit
        )
        self.assertEqual(results.solver.deterministic_time, 100.00)

    def test_log_file_shows_max_deterministic_time_limit_exceeded_with_feasible_solution(self):
        log_file_text = """
MIP - Deterministic time limit exceeded, integer feasible:  Objective =  0.0000000000e+00
Current MIP best bound =  0.0000000000e+00 (gap = 10.0, 10.00%)
Solution time =   10.00 sec.  Iterations = 10000  Nodes = 1000 (1)
Deterministic time = 100.00 ticks  (10.00 ticks/sec)

CPLEX> Incumbent solution written to file '/var/folders/_x/xxxx/T/tmpxxx.cplex.sol'.
CPLEX>"""
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.status, SolverStatus.ok)
        self.assertEqual(
            results.solver.termination_condition, TerminationCondition.maxTimeLimit
        )
        self.assertEqual(results.solver.deterministic_time, 100.00)

    def test_log_file_shows_warm_start_objective_value(self):
        log_file_text = """
1 of 1 MIP starts provided solutions.
MIP start 'm1' defined initial solution with objective 25210.5363.
"""
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.warm_start_objective_value, 25210.5363)

    def test_log_file_shows_root_node_processing_time(self):
        log_file_text = """
Presolve time = 0.14 sec. (181.11 ticks)

Root node processing (before b&c):
  Real time             =    123.45 sec. (211.39 ticks)
Parallel b&c, 16 threads:
  Real time             =    67.89 sec. (56.98 ticks)
  Sync time (average)   =    0.00 sec.
  Wait time (average)   =    0.00 sec.
                          ------------
Total (root+branch&cut) =    191.34 sec. (268.37 ticks)
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.root_node_processing_time, 123.45)

    def test_log_file_shows_tree_processing_time_when_parallel(self):
        log_file_text = """
Presolve time = 0.14 sec. (181.11 ticks)

Root node processing (before b&c):
  Real time             =    123.45 sec. (211.39 ticks)
Parallel b&c, 16 threads:
  Real time             =    67.89 sec. (56.98 ticks)
  Sync time (average)   =    0.00 sec.
  Wait time (average)   =    0.00 sec.
                          ------------
Total (root+branch&cut) =    191.34 sec. (268.37 ticks)
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.tree_processing_time, 67.89)

    def test_log_file_shows_tree_processing_time_when_sequential(self):
        log_file_text = """
Presolve time = 0.14 sec. (181.11 ticks)

Root node processing (before b&c):
  Real time             =    123.45 sec. (211.39 ticks)
Sequential b&c:
  Real time             =    67.89 sec. (56.98 ticks)
  Sync time (average)   =    0.00 sec.
  Wait time (average)   =    0.00 sec.
                          ------------
Total (root+branch&cut) =    191.34 sec. (268.37 ticks)
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.tree_processing_time, 67.89)

    def test_log_file_shows_n_solutions_found_when_multiple(self):
        log_file_text = """
Solution pool: 15 solutions saved.
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.n_solutions_found, 15)

    def test_log_file_shows_n_solutions_found_when_single(self):
        log_file_text = """
Solution pool: 1 solution saved.
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.solver.n_solutions_found, 1)

    def test_log_file_shows_number_of_binary_variables(self):
        log_file_text = """
Objective sense      : Minimize
Variables            :     506  [Nneg: 206,  Binary: 300]
Objective nonzeros   :      32
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.problem.number_of_binary_variables, 300)

    def test_log_file_shows_number_of_binary_variables_when_integer_variables_are_present(
        self
    ):
        log_file_text = """
Variables : 7 [Nneg: 1, Binary: 4, General Integer: 2]
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.problem.number_of_binary_variables, 4)

    def test_log_file_shows_number_of_continuous_variables(self):
        log_file_text = """
Objective sense      : Minimize
Variables            :     506  [Nneg: 206,  Binary: 300]
Objective nonzeros   :      32
 """
        with open(self.solver._log_file, "w") as f:
            f.write(log_file_text)

        results = CPLEXSHELL.process_logfile(self.solver)
        self.assertEqual(results.problem.number_of_continuous_variables, 206)


if __name__ == "__main__":
    unittest.main()
