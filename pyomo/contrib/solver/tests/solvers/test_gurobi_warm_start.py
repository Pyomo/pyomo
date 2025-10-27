#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import logging
import math
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import TerminationCondition, SolutionStatus
from pyomo.contrib.solver.solvers.gurobi_direct_minlp import GurobiDirectMINLP

from pyomo.core.base.constraint import Constraint
from pyomo.environ import (
    ConcreteModel,
    Var,
    Constraint,
    value,
    Binary,
    NonNegativeReals,
    Objective,
    Set,
)

gurobi_direct = SolverFactory('gurobi_direct')


@unittest.skipUnless(gurobi_direct.available(), "needs Gurobi Direct interface")
class TestGurobiDirect(unittest.TestCase):
    def test_warm_start(self):
        m = ConcreteModel()
        m.S = Set(initialize=[1, 2, 3, 4, 5])
        m.y = Var(m.S, domain=Binary)
        m.x = Var(m.S, domain=NonNegativeReals)
        m.obj = Objective(expr=sum(m.x[i] for i in m.S))

        @m.Constraint(m.S)
        def cons(m, i):
            if i % 2 == 0:
                return m.x[i] + i * m.y[i] >= 3 * i
            else:
                return m.x[i] - i * m.y[i] >= 3 * i

        # define a suboptimal MIP start
        for i in m.S:
            m.y[i] = 1
        # objective will be 4 + 4 + 12 + 8 + 20 = 48

        gurobi_direct.config.warm_start = True
        logger = logging.getLogger('tee')
        with LoggingIntercept(module='tee', level=logging.INFO) as LOG:
            gurobi_direct.solve(m, tee=logger)
        self.assertIn(
            "User MIP start produced solution with objective 48", LOG.getvalue()
        )

        # check that we got the optimal solution:
        # y[1] = 0, x[1] = 3
        # y[2] = 1, x[2] = 4
        # y[3] = 0, x[3] = 9
        # y[4] = 1, x[4] = 8
        # y[5] = 0, x[5] = 15
        x = {1: 3, 2: 4, 3: 9, 4: 8, 5: 15}
        self.assertEqual(value(m.obj), 39)
        for i in m.S:
            if i % 2 == 0:
                self.assertEqual(value(m.y[i]), 1)
            else:
                self.assertEqual(value(m.y[i]), 0)
            self.assertEqual(value(m.x[i]), x[i])
