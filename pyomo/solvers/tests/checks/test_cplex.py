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

from pyomo.core import Binary, ConcreteModel, Constraint, Objective, Var
from pyomo.opt import ProblemFormat, convert_problem
from pyomo.solvers.plugins.solvers.CPLEX import (CPLEXSHELL, MockCPLEX,
                                                 _validate_file_name)


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


class CPLEXShellPrioritiesFile(unittest.TestCase):
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

    def test_write_empty_priorities_file(self):
        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

        with open(self.mock_cplex_shell._priorities_file_name, "r") as ord_file:
            priorities_file = ord_file.read()

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\nENDATA\n",
        )

    def test_write_priority_to_priorities_file(self):
        self.mock_model.x.branch_priority = 10

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x1 10\nENDATA\n",
        )
        self.assertIsNone(self.mock_model.x.branch_direction)

    def test_write_priority_and_direction_to_priorities_file(self):
        self.mock_model.x.branch_priority = 10
        self.mock_model.x.branch_direction = -1

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\n DN x1 10\nENDATA\n",
        )

    def test_raise_due_to_invalid_priority(self):
        self.mock_model.x.branch_priority = -1
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

        self.mock_model.x.branch_priority = 1.1
        with self.assertRaises(ValueError):
            CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)

    def test_use_default_due_to_invalid_direction(self):
        self.mock_model.x.branch_priority = 10
        self.mock_model.x.branch_direction = "invalid_branching_direction"

        CPLEXSHELL._write_priorities_file(self.mock_cplex_shell, self.mock_model)
        priorities_file = self.get_priorities_file_as_string(self.mock_cplex_shell)

        self.assertEqual(
            priorities_file,
            "* ENCODING=ISO-8859-1\nNAME             Priority Order\n  x1 10\nENDATA\n",
        )


if __name__ == "__main__":
    unittest.main()
