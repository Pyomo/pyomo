# ____________________________________________________________________________________
#
# Pyomo: Python Optimization Modeling Objects
# Copyright (c) 2008-2026 National Technology and Engineering Solutions of Sandia, LLC
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this
# software.  This software is distributed under the 3-clause BSD License.
# ____________________________________________________________________________________

import logging
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.common.factory import SolverFactory

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
gurobi_direct_minlp = SolverFactory('gurobi_direct_minlp')
gurobi_persistent = SolverFactory('gurobi_persistent')


class TestGurobiWarmStart(unittest.TestCase):
    def make_model(self):
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

        return m

    def check_optimal_soln(self, m):
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

    @unittest.skipUnless(gurobi_direct.available(), "needs Gurobi Direct interface")
    @unittest.pytest.mark.solver("gurobi_direct")
    def test_gurobi_direct_warm_start(self):
        m = self.make_model()

        gurobi_direct.config.warmstart_discrete_vars = True
        logger = logging.getLogger('tee')
        with LoggingIntercept(module='tee', level=logging.INFO) as LOG:
            gurobi_direct.solve(m, tee=logger)
        self.assertIn(
            "User MIP start produced solution with objective 48", LOG.getvalue()
        )
        self.check_optimal_soln(m)

    @unittest.skipUnless(
        gurobi_direct_minlp.available(), "needs Gurobi Direct MINLP interface"
    )
    @unittest.pytest.mark.solver("gurobi_direct_minlp")
    def test_gurobi_minlp_warmstart(self):
        m = self.make_model()

        gurobi_direct_minlp.config.warmstart_discrete_vars = True
        logger = logging.getLogger('tee')
        with LoggingIntercept(module='tee', level=logging.INFO) as LOG:
            gurobi_direct_minlp.solve(m, tee=logger)
        self.assertIn(
            "User MIP start produced solution with objective 48", LOG.getvalue()
        )
        self.check_optimal_soln(m)

    @unittest.skipUnless(
        gurobi_persistent.available(), "needs Gurobi persistent interface"
    )
    @unittest.pytest.mark.solver("gurobi_persistent")
    def test_gurobi_persistent_warmstart(self):
        m = self.make_model()

        gurobi_persistent.config.warmstart_discrete_vars = True
        gurobi_persistent.set_instance(m)
        logger = logging.getLogger('tee')
        with LoggingIntercept(module='tee', level=logging.INFO) as LOG:
            gurobi_persistent.solve(m, tee=logger)
        self.assertIn(
            "User MIP start produced solution with objective 48", LOG.getvalue()
        )
        self.check_optimal_soln(m)
