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

from pyomo.common.tempfiles import TempfileManager
import pyutilib.th as unittest

import pyomo.kernel as pmo
from pyomo.core import Binary, ConcreteModel, Constraint, Objective, Var, Integers, RangeSet, minimize, quicksum, Suffix
from pyomo.opt import ProblemFormat, convert_problem, SolverFactory, BranchDirection
from pyomo.solvers.plugins.solvers.CPLEX import CPLEXSHELL, MockCPLEX, _validate_file_name


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
    """ Unit test on writing of priorities via `CPLEXSHELL._write_priorities_file()` """
    suffix_cls = Suffix

    def setUp(self):
        self.mock_model = self.get_mock_model()
        self.mock_cplex_shell = self.get_mock_cplex_shell(self.mock_model)
        self.mock_cplex_shell._priorities_file_name = TempfileManager.create_tempfile(
            suffix=".cplex.ord"
        )

    def tearDown(self):
        TempfileManager.clear_tempfiles()

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

    @staticmethod
    def _set_suffix_value(suffix, variable, value):
        suffix.set_value(variable, value)

    def test_write_without_priority_suffix(self):
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_write_priority_to_priorities_file(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\n"
            "NAME             Priority Order\n"
            "  x1 10\n"
            "ENDATA\n"
        )

    def test_write_priority_and_direction_to_priorities_file(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)

        self.mock_model.direction = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        direction_val = BranchDirection.down
        self._set_suffix_value(self.mock_model.direction, self.mock_model.x, direction_val)

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\n"
            "NAME             Priority Order\n"
            " DN x1 10\n"
            "ENDATA\n"
        )

    def test_raise_due_to_invalid_priority(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, -1)
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, 1.1)
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_use_default_due_to_invalid_direction(self):
        self.mock_model.priority = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        priority_val = 10
        self._set_suffix_value(self.mock_model.priority, self.mock_model.x, priority_val)

        self.mock_model.direction = self.suffix_cls(direction=Suffix.EXPORT, datatype=Suffix.INT)
        self._set_suffix_value(
            self.mock_model.direction, self.mock_model.x, "invalid_branching_direction"
        )

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\n"
            "NAME             Priority Order\n"
            "  x1 10\n"
            "ENDATA\n"
        )


class CPLEXShellWritePrioritiesFileKernel(CPLEXShellWritePrioritiesFile):
    suffix_cls = pmo.suffix

    @staticmethod
    def _set_suffix_value(suffix, variable, value):
        suffix[variable] = value

    def get_mock_model(self):
        model = pmo.block()
        model.x = pmo.variable(domain=Binary)
        model.con = pmo.constraint(expr=model.x >= 1)
        model.obj = pmo.objective(expr=model.x)
        return model


class CPLEXShellSolvePrioritiesFile(unittest.TestCase):
    """ Integration test on the end-to-end application of priorities via the `Suffix` through a `solve()` """
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

        self.assertEqual(
            priorities_file,
            (
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
            ),
        )
        self.assertIn("read %s\n" % (opt._priorities_file_name,), opt._command.script)

    def test_ignore_variable_priorities(self):
        model = self.get_mock_model_with_priorities()
        with SolverFactory("_mock_cplex") as opt:
            opt._presolve(model, priorities=False, keepfiles=True)

            self.assertIsNone(opt._priorities_file_name)
            self.assertNotIn(".ord", opt._command.script)

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

            self.assertIn(".ord", opt._command.script)

            with open(opt._priorities_file_name, "r") as ord_file:
                priorities_file = ord_file.read()

        self.assertEqual(priorities_file, provided_priorities_file)


class CPLEXShellSolvePrioritiesFileKernel(CPLEXShellSolvePrioritiesFile):
    def get_mock_model_with_priorities(self):
        m = pmo.block()
        m.x = pmo.variable(domain=Integers)
        m.s = range(10)

        m.y = pmo.variable_list(pmo.variable(domain=Integers) for _ in m.s)

        m.o = pmo.objective(expr=m.x + sum(m.y), sense=minimize)
        m.c = pmo.constraint(expr=m.x >= 1)
        m.c2 = pmo.constraint(expr=quicksum(m.y[i] for i in m.s) >= 10)

        m.priority = pmo.suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)
        m.direction = pmo.suffix(direction=Suffix.EXPORT, datatype=Suffix.INT)

        m.priority[m.x] = 1
        m.priority[m.y] = 2
        m.direction[m.y] = BranchDirection.down
        m.direction[m.y[-1]] = BranchDirection.up
        return m


if __name__ == "__main__":
    unittest.main()
