from os.path import abspath, dirname, join, normpath

import pyutilib.th as unittest
from pyutilib.misc import import_file

from pyomo.contrib.satsolver.satsolver import satisfiable, z3_available
from pyomo.core.base.set_types import PositiveIntegers, NonNegativeReals, Binary
from pyomo.environ import (
    ConcreteModel, Var, Constraint, Objective, sin, cos, tan, asin, acos, atan, sqrt, log,
    minimize)
from pyomo.gdp import Disjunct, Disjunction

currdir = dirname(abspath(__file__))
exdir = normpath(join(currdir, '..', '..', '..', 'examples', 'gdp'))


@unittest.skipUnless(z3_available, "Z3 SAT solver is not available.")
class SatSolverTests(unittest.TestCase):

    def test_simple_sat_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr=1 == m.x)
        m.o = Objective(expr=m.x)
        self.assertTrue(satisfiable(m))

    def test_simple_unsat_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr=1 == m.x)
        m.c2 = Constraint(expr=2 == m.x)
        m.o = Objective(expr=m.x)
        self.assertFalse(satisfiable(m))

    def test_bounds_sat(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 5))
        m.c1 = Constraint(expr=4.99 == m.x)
        m.o = Objective(expr=m.x)
        self.assertTrue(satisfiable(m))

    def test_upper_bound_unsat(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 5))
        m.c = Constraint(expr=5.01 == m.x)
        m.o = Objective(expr=m.x)
        self.assertFalse(satisfiable(m))

    def test_lower_bound_unsat(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 5))
        m.c = Constraint(expr=-0.01 == m.x)
        m.o = Objective(expr=m.x)
        self.assertFalse(satisfiable(m))

    def test_binary_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(expr=0 <= m.x + m.y - m.z * m.y / m.x + 7)
        m.o = Objective(expr=m.x)
        self.assertTrue(satisfiable(m))

    def test_unary_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        m.c1 = Constraint(expr=0 <= sin(m.x))
        m.c2 = Constraint(expr=0 <= cos(m.y))
        m.c3 = Constraint(expr=0 <= tan(m.z))
        m.c4 = Constraint(expr=0 <= asin(m.a))
        m.c5 = Constraint(expr=0 <= acos(m.b))
        m.c6 = Constraint(expr=0 <= atan(m.c))
        m.c7 = Constraint(expr=0 <= sqrt(m.d))
        m.o = Objective(expr=m.x)
        self.assertTrue(satisfiable(m) is not False)

    def test_unhandled_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr=0 <= log(m.x))
        self.assertTrue(satisfiable(m))

    def test_abs_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr=-0.001 >= abs(m.x))
        m.o = Objective(expr=m.x)
        self.assertFalse(satisfiable(m))

    def test_inactive_constraints(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr=m.x == 1)
        m.c2 = Constraint(expr=m.x == 2)
        m.o = Objective(expr=m.x)
        self.assertFalse(satisfiable(m))
        m.c2.deactivate()
        self.assertTrue(satisfiable(m))

    def test_disjunction_sat1(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 8))
        m.x2 = Var(bounds=(0, 8))
        m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x1 >= 2)
        m.y1.c2 = Constraint(expr=m.x2 >= 2)
        m.y2.c1 = Constraint(expr=m.x1 >= 9)
        m.y2.c2 = Constraint(expr=m.x2 >= 3)
        m.djn = Disjunction(expr=[m.y1, m.y2])
        self.assertTrue(satisfiable(m))

    def test_disjunction_sat1(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 8))
        m.x2 = Var(bounds=(0, 8))
        m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x1 >= 9)
        m.y1.c2 = Constraint(expr=m.x2 >= 2)
        m.y2.c1 = Constraint(expr=m.x1 >= 3)
        m.y2.c2 = Constraint(expr=m.x2 >= 3)
        m.djn = Disjunction(expr=[m.y1, m.y2])
        self.assertTrue(satisfiable(m))

    def test_disjunction_unsat(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 8))
        m.x2 = Var(bounds=(0, 8))
        m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x1 >= 9)
        m.y1.c2 = Constraint(expr=m.x2 >= 2)
        m.y2.c1 = Constraint(expr=m.x1 >= 3)
        m.y2.c2 = Constraint(expr=m.x2 >= 9)
        m.djn = Disjunction(expr=[m.y1, m.y2])
        self.assertFalse(satisfiable(m))

    def test_multiple_disjunctions_unsat(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 8))
        m.x2 = Var(bounds=(0, 8))
        m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x1 >= 2)
        m.y1.c2 = Constraint(expr=m.x2 >= 2)
        m.y2.c1 = Constraint(expr=m.x1 >= 2)
        m.y2.c2 = Constraint(expr=m.x2 >= 2)
        m.djn1 = Disjunction(expr=[m.y1, m.y2])
        m.z1 = Disjunct()
        m.z2 = Disjunct()
        m.z1.c1 = Constraint(expr=m.x1 <= 1)
        m.z1.c2 = Constraint(expr=m.x2 <= 1)
        m.z2.c1 = Constraint(expr=m.x1 <= 1)
        m.z2.c2 = Constraint(expr=m.x2 <= 1)
        m.djn2 = Disjunction(expr=[m.z1, m.z2])
        self.assertFalse(satisfiable(m))

    def test_multiple_disjunctions_sat(self):
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 8))
        m.x2 = Var(bounds=(0, 8))
        m.obj = Objective(expr=m.x1 + m.x2, sense=minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr=m.x1 >= 2)
        m.y1.c2 = Constraint(expr=m.x2 >= 2)
        m.y2.c1 = Constraint(expr=m.x1 >= 1)
        m.y2.c2 = Constraint(expr=m.x2 >= 1)
        m.djn1 = Disjunction(expr=[m.y1, m.y2])
        m.z1 = Disjunct()
        m.z2 = Disjunct()
        m.z1.c1 = Constraint(expr=m.x1 <= 1)
        m.z1.c2 = Constraint(expr=m.x2 <= 1)
        m.z2.c1 = Constraint(expr=m.x1 <= 0)
        m.z2.c2 = Constraint(expr=m.x2 <= 0)
        m.djn2 = Disjunction(expr=[m.z1, m.z2])
        self.assertTrue(satisfiable(m))

    def test_integer_domains(self):
        m = ConcreteModel()
        m.x1 = Var(domain=PositiveIntegers)
        m.c1 = Constraint(expr=m.x1 == 0.5)
        self.assertFalse(satisfiable(m))

    def test_real_domains(self):
        m = ConcreteModel()
        m.x1 = Var(domain=NonNegativeReals)
        m.c1 = Constraint(expr=m.x1 == -1.3)
        self.assertFalse(satisfiable(m))

    def test_binary_domains(self):
        m = ConcreteModel()
        m.x1 = Var(domain=Binary)
        m.c1 = Constraint(expr=m.x1 == 2)
        self.assertFalse(satisfiable(m))

    def test_8PP(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        m = exfile.build_eight_process_flowsheet()
        self.assertTrue(satisfiable(m) is not False)

    def test_8PP_deactive(self):
        exfile = import_file(
            join(exdir, 'eight_process', 'eight_proc_model.py'))
        m = exfile.build_eight_process_flowsheet()
        for djn in m.component_data_objects(ctype=Disjunction):
            djn.deactivate()
        self.assertTrue(satisfiable(m) is not False)

    def test_strip_pack(self):
        exfile = import_file(
            join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        m = exfile.build_rect_strip_packing_model()
        self.assertTrue(satisfiable(m))

    def test_constrained_layout(self):
        exfile = import_file(
            join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        m = exfile.build_constrained_layout_model()
        self.assertTrue(satisfiable(m) is not False)

    def test_ex_633_trespalacios(self):
        exfile = import_file(join(exdir, 'small_lit', 'ex_633_trespalacios.py'))
        m = exfile.build_simple_nonconvex_gdp()
        self.assertTrue(satisfiable(m) is not False)


if __name__ == '__main__':
    unittest.main()
