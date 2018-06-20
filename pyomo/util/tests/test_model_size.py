"""Tests for the model size report utility."""
import logging
from os.path import abspath, dirname, join, normpath

from six import StringIO

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core import Binary, Block, ConcreteModel, Constraint, Integers, Var
from pyomo.gdp import Disjunct, Disjunction
from pyomo.util.model_size import (build_model_size_report,
                                   log_model_size_report)
from pyutilib.misc import import_file

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'gdp'))


class TestModelSizeReport(unittest.TestCase):
    """Tests for model size report utility."""

    def test_empty_model(self):
        """Test with an empty model."""
        empty_model = ConcreteModel()
        model_size = build_model_size_report(empty_model)
        for obj in model_size.active.values():
            self.assertEqual(obj, 0)

    @unittest.skip("Example file is not where it should be yet.")
    def test_eight_process(self):
        """Test with the eight process problem model."""
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        model_size = build_model_size_report(eight_process)
        self.assertEqual(model_size.active.variables, 30)  # 30 is placeholder

    def test_nested_blocks(self):
        """Test with nested blocks."""
        m = ConcreteModel()
        m.b = Block()
        m.inactive_b = Block()
        m.inactive_b.deactivate()
        m.b.x = Var()
        m.b.x2 = Var(domain=Binary)
        m.b.x3 = Var(domain=Integers)
        m.inactive_b.x = Var()
        m.b.c = Constraint(expr=m.b.x == m.b.x2)
        m.inactive_b.c = Constraint(expr=m.b.x == 1)
        m.inactive_b.c2 = Constraint(expr=m.inactive_b.x == 15)
        model_size = build_model_size_report(m)
        self.assertEqual(model_size.active.variables, 2)
        self.assertEqual(model_size.overall.variables, 4)
        self.assertEqual(model_size.active.binary_variables, 1)
        self.assertEqual(model_size.overall.binary_variables, 1)
        self.assertEqual(model_size.active.integer_variables, 0)
        self.assertEqual(model_size.overall.integer_variables, 1)
        self.assertEqual(model_size.active.constraints, 1)
        self.assertEqual(model_size.overall.constraints, 3)
        self.assertEqual(model_size.active.disjuncts, 0)
        self.assertEqual(model_size.overall.disjuncts, 0)
        self.assertEqual(model_size.active.disjunctions, 0)
        self.assertEqual(model_size.overall.disjunctions, 0)

    def test_disjunctive_model(self):
        from pyomo.gdp.tests.models import makeNestedDisjunctions
        m = makeNestedDisjunctions()
        model_size = build_model_size_report(m)
        self.assertEqual(model_size.active.variables, 9)
        self.assertEqual(model_size.overall.variables, 10)
        self.assertEqual(model_size.active.binary_variables, 7)
        self.assertEqual(model_size.overall.binary_variables, 7)
        self.assertEqual(model_size.active.integer_variables, 0)
        self.assertEqual(model_size.overall.integer_variables, 0)
        self.assertEqual(model_size.active.constraints, 6)
        self.assertEqual(model_size.overall.constraints, 6)
        self.assertEqual(model_size.active.disjuncts, 7)
        self.assertEqual(model_size.overall.disjuncts, 7)
        self.assertEqual(model_size.active.disjunctions, 3)
        self.assertEqual(model_size.overall.disjunctions, 3)

    def test_nonlinear(self):
        """Test nonlinear constraint detection."""
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.z.fix(3)
        m.c = Constraint(expr=m.x ** 2 == 4)
        m.c2 = Constraint(expr=m.x / m.y == 3)
        m.c3 = Constraint(expr=m.x * m.z == 5)
        m.c4 = Constraint(expr=m.x * m.y == 5)
        m.c4.deactivate()
        model_size = build_model_size_report(m)
        self.assertEqual(model_size.active.nonlinear_constraints, 2)
        self.assertEqual(model_size.overall.nonlinear_constraints, 3)

    def test_unassociated_disjunct(self):
        m = ConcreteModel()
        m.x = Var(domain=Integers)
        m.d = Disjunct()
        m.d.c = Constraint(expr=m.x == 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x == 5)
        m.disj = Disjunction(expr=[m.d2])
        model_size = build_model_size_report(m)
        self.assertEqual(model_size.warn.unassociated_disjuncts, 1)

    def test_log_model_size(self):
        """Test logging functionality."""
        m = ConcreteModel()
        m.x = Var(domain=Integers)
        m.d = Disjunct()
        m.d.c = Constraint(expr=m.x == 1)
        m.d2 = Disjunct()
        m.d2.c = Constraint(expr=m.x == 5)
        m.disj = Disjunction(expr=[m.d2])
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.util.model_size', logging.INFO):
            log_model_size_report(m)
        expected_output = """
active:
    binary_variables: 1
    constraints: 1
    continuous_variables: 0
    disjunctions: 1
    disjuncts: 1
    integer_variables: 1
    nonlinear_constraints: 0
    variables: 2
overall:
    binary_variables: 2
    constraints: 2
    continuous_variables: 0
    disjunctions: 1
    disjuncts: 2
    integer_variables: 1
    nonlinear_constraints: 0
    variables: 3
warn:
    unassociated_disjuncts: 1
        """.strip()
        self.assertEqual(output.getvalue().strip(), expected_output)


if __name__ == '__main__':
    unittest.main()
