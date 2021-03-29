import pyutilib.th as unittest
import pyomo.environ as pe
from pyomo.contrib import appsi
import os


class TestNLWriter(unittest.TestCase):
    def test_header_1(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '0 0',
                         '0 0',
                         '0 0 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_2(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '0 1',
                         '0 0',
                         '0 1 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_3(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 0',
                         '0 0',
                         '1 0 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_4(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '1 1 1',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_5(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '1 2 1',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_6(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '2 1 1',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_7(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 0',
                         '0 0',
                         '1 0 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_8(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 0',
                         '0 0',
                         '2 0 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_9(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '0 1',
                         '0 0',
                         '0 1 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_10(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '1 1 1',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_11(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '1 2 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_12(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '2 1 1',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_13(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '1 2 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_14(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '0 1',
                         '0 0',
                         '0 2 0',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_15(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '1 2 1',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')

    def test_header_16(self):
        m = pe.ConcreteModel()
        m.x = pe.Var()
        m.y = pe.Var()
        m.obj = pe.Objective(expr=m.x**2 + m.y**2)
        m.c = pe.Constraint(expr=m.x**2 + m.y**2 == 1)
        writer = appsi.writers.NLWriter()
        writer.write(m, 'tmp.nl')
        correct_lines = ['g3 1 1 0',
                         '2 1 1 0 1',
                         '1 1',
                         '0 0',
                         '2 2 2',
                         '0 0 0 1',
                         '0 0 0 0 0',
                         '2 2',
                         '0 0',
                         '0 0 0 0 0']
        f = open('tmp.nl', 'r')
        for ndx, line in enumerate(list(f.readlines())[:10]):
            self.assertTrue(line.startswith(correct_lines[ndx]))
        f.close()
        os.remove('tmp.nl')
