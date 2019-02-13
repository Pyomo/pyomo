import logging

from six import StringIO
from six.moves import range

import pyutilib.th as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.contrib.satsolver import SMTSatSolver
from pyomo.environ import *


if __name__ == "__main__":
    m = ConcreteModel()
    m.x = Var()
    m.z = Var()
    m.c1 = Constraint(expr= 1 == (m.x))
    m.c2 = Constraint(expr= 1 <= 1 + (m.x))
    m.o = Objective(expr=m.x*m.z)
    smt_model = SMTSatSolver(model = m)
    print smt_model.check()
