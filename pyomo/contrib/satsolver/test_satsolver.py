import logging

from six import StringIO
from six.moves import range
import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.satsolver.satsolver import SMTSatSolver
from pyomo.environ import *
from pyomo.gdp import Disjunct, Disjunction

class SatSolverTests(unittest.TestCase):

    def test_simple_sat_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.c = Constraint(expr= 1 == (m.x))
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")


    def test_simple_unsat_model(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr= 1 == (m.x))
        m.c2 = Constraint(expr= 2 == (m.x))
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")


    def test_bounds_sat(self):
        m = ConcreteModel()
        m.x = Var(bounds = (0,5))
        m.c1 = Constraint(expr= 4.99 == (m.x))
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")


    def test_upper_bound_unsat(self):
        m = ConcreteModel()
        m.x = Var(bounds = (0,5))
        m.c = Constraint(expr= 5.01 == (m.x))
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")


    def test_lower_bound_unsat(self):
        m = ConcreteModel()
        m.x = Var(bounds = (0,5))
        m.c = Constraint(expr= -0.01 == (m.x))
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")


    def test_binary_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.c = Constraint(expr=0 <= m.x + m.y - m.z * m.y / m.x + 7  )
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")


    def test_unary_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.y = Var()
        m.z = Var()
        m.a = Var()
        m.b = Var()
        m.c = Var()
        m.d = Var()
        m.c1 = Constraint(expr=0 <= sin(m.x) )
        m.c2 = Constraint(expr=0 <= cos(m.y) )
        m.c3 = Constraint(expr=0 <= tan(m.z) )
        m.c4 = Constraint(expr=0 <= asin(m.a) )
        m.c5 = Constraint(expr=0 <= acos(m.b) )
        m.c6 = Constraint(expr=0 <= atan(m.c) )
        m.c7 = Constraint(expr=0 <= sqrt(m.d) )
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertFalse(str(smt_model.check()) =="unsat")

    def test_abs_expressions(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr=-0.001 >= abs(m.x) )
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")

    def test_inactive_constraints(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr= m.x==1 )
        m.c2 = Constraint(expr= m.x==2 )
        m.o = Objective(expr=m.x)
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")
        m.c2.deactivate()
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")

    def test_disjunction_sat1(self):
        m = ConcreteModel()
        m.x1 = Var(bounds = (0,8))
        m.x2 = Var(bounds = (0,8))
        m.obj = Objective(expr=m.x1 + m.x2,sense = minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr = m.x1 >= 2)
        m.y1.c2 = Constraint(expr = m.x2 >= 2)
        m.y2.c1 = Constraint(expr = m.x1 >= 9)
        m.y2.c2 = Constraint(expr = m.x2 >= 3)
        m.djn = Disjunction(expr=[m.y1,m.y2])
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")

    def test_disjunction_sat1(self):
        m = ConcreteModel()
        m.x1 = Var(bounds = (0,8))
        m.x2 = Var(bounds = (0,8))
        m.obj = Objective(expr=m.x1 + m.x2,sense = minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr = m.x1 >= 9)
        m.y1.c2 = Constraint(expr = m.x2 >= 2)
        m.y2.c1 = Constraint(expr = m.x1 >= 3)
        m.y2.c2 = Constraint(expr = m.x2 >= 3)
        m.djn = Disjunction(expr=[m.y1,m.y2])
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")

    def test_disjunction_unsat(self):
        m = ConcreteModel()
        m.x1 = Var(bounds = (0,8))
        m.x2 = Var(bounds = (0,8))
        m.obj = Objective(expr=m.x1 + m.x2,sense = minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr = m.x1 >= 9)
        m.y1.c2 = Constraint(expr = m.x2 >= 2)
        m.y2.c1 = Constraint(expr = m.x1 >= 3)
        m.y2.c2 = Constraint(expr = m.x2 >= 9)
        m.djn = Disjunction(expr=[m.y1,m.y2])
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")

    def test_multiple_disjunctions_unsat(self):
        m = ConcreteModel()
        m.x1 = Var(bounds = (0,8))
        m.x2 = Var(bounds = (0,8))
        m.obj = Objective(expr=m.x1 + m.x2,sense = minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr = m.x1 >= 2)
        m.y1.c2 = Constraint(expr = m.x2 >= 2)
        m.y2.c1 = Constraint(expr = m.x1 >= 2)
        m.y2.c2 = Constraint(expr = m.x2 >= 2)
        m.djn1 = Disjunction(expr=[m.y1,m.y2])
        m.z1 = Disjunct()
        m.z2 = Disjunct()
        m.z1.c1 = Constraint(expr = m.x1 <= 1)
        m.z1.c2 = Constraint(expr = m.x2 <= 1)
        m.z2.c1 = Constraint(expr = m.x1 <= 1)
        m.z2.c2 = Constraint(expr = m.x2 <= 1)
        m.djn2 = Disjunction(expr=[m.z1,m.z2])
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="unsat")

    def test_multiple_disjunctions_sat(self):
        m = ConcreteModel()
        m.x1 = Var(bounds = (0,8))
        m.x2 = Var(bounds = (0,8))
        m.obj = Objective(expr=m.x1 + m.x2,sense = minimize)
        m.y1 = Disjunct()
        m.y2 = Disjunct()
        m.y1.c1 = Constraint(expr = m.x1 >= 2)
        m.y1.c2 = Constraint(expr = m.x2 >= 2)
        m.y2.c1 = Constraint(expr = m.x1 >= 1)
        m.y2.c2 = Constraint(expr = m.x2 >= 1)
        m.djn1 = Disjunction(expr=[m.y1,m.y2])
        m.z1 = Disjunct()
        m.z2 = Disjunct()
        m.z1.c1 = Constraint(expr = m.x1 <= 1)
        m.z1.c2 = Constraint(expr = m.x2 <= 1)
        m.z2.c1 = Constraint(expr = m.x1 <= 0)
        m.z2.c2 = Constraint(expr = m.x2 <= 0)
        m.djn2 = Disjunction(expr=[m.z1,m.z2])
        smt_model = SMTSatSolver(model = m)
        self.assertTrue(str(smt_model.check()) =="sat")



if __name__ == '__main__':
    unittest.main()
