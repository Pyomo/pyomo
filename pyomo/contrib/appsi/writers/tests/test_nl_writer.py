import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.environ as pe
from pyomo.contrib import appsi
from pyomo.contrib.appsi.cmodel import cmodel_available
import os


@unittest.skipUnless(cmodel_available, 'appsi extensions are not available')
class TestNLWriter(unittest.TestCase):
    def test_all_vars_fixed(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c1 = pe.Constraint(expr=m.y >= pe.exp(m.x))
        m.c2 = pe.Constraint(expr=m.y >= (m.x - 1) ** 2)
        m.x.fix(1)
        m.y.fix(2)
        writer = appsi.writers.NLWriter()
        with TempfileManager:
            fname = TempfileManager.create_tempfile(suffix='.appsi.nl')
            with self.assertRaisesRegex(
                ValueError, 'there are not any unfixed variables in the problem'
            ):
                writer.write(m, fname)

    def _write_and_check_header(self, m, correct_lines):
        writer = appsi.writers.NLWriter()
        with TempfileManager:
            fname = TempfileManager.create_tempfile(suffix='.appsi.nl')
            writer.write(m, fname)
            with open(fname, 'r') as f:
                for ndx, line in enumerate(list(f.readlines())[:10]):
                    self.assertTrue(line.startswith(correct_lines[ndx]))

    def test_header_1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '0 0',
            '0 0',
            '0 0 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '0 1',
            '0 0',
            '0 1 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_3(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 0',
            '0 0',
            '1 0 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_4(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '1 1 1',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_5(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '1 2 1',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_6(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '2 1 1',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_7(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 0',
            '0 0',
            '1 0 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_8(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 0',
            '0 0',
            '2 0 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_9(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '0 1',
            '0 0',
            '0 1 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_10(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '1 1 1',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_11(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '1 2 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_12(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '2 1 1',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_13(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '1 2 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_14(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '0 1',
            '0 0',
            '0 2 0',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_15(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '1 2 1',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)

    def test_header_16(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        correct_lines = [
            'g3 1 1 0',
            '2 1 1 0 1',
            '1 1',
            '0 0',
            '2 2 2',
            '0 0 0 1',
            '0 0 0 0 0',
            '2 2',
            '0 0',
            '0 0 0 0 0',
        ]
        self._write_and_check_header(m, correct_lines)
