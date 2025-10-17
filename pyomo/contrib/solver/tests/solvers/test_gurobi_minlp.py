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

import math
import pyomo.common.unittest as unittest
from pyomo.contrib.solver.common.factory import SolverFactory
from pyomo.contrib.solver.common.results import TerminationCondition, SolutionStatus
from pyomo.contrib.solver.solvers.gurobi_direct_minlp import GurobiDirectMINLP

from pyomo.core.base.constraint import Constraint
from pyomo.environ import (
    ConcreteModel,
    Var,
    VarList,
    Constraint,
    ConstraintList,
    value,
    Binary,
    NonNegativeReals,
    Objective,
    maximize,
    minimize,
    cos,
    sin,
    tan,
    log,
    log10,
    exp,
    sqrt,
)

gurobi_direct = SolverFactory('gurobi_direct_minlp')


@unittest.skipUnless(gurobi_direct.available(), "needs Gurobi Direct MINLP interface")
class TestGurobiMINLP(unittest.TestCase):
    def test_gurobi_minlp_sincosexp(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(-1, 4))
        m.o = Objective(expr=sin(m.x) + cos(2 * m.x) + 1)
        m.c = Constraint(expr=0.25 * exp(m.x) - m.x <= 0)
        gurobi_direct.solve(m)
        self.assertAlmostEqual(1, value(m.o), delta=1e-3)
        self.assertAlmostEqual(1.571, value(m.x), delta=1e-3)

    def test_gurobi_minlp_tan(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(0, math.pi / 2))
        m.o = Objective(expr=tan(m.x) / (m.x**2))
        gurobi_direct.solve(m)
        self.assertAlmostEqual(0.948, value(m.x), delta=1e-3)
        self.assertAlmostEqual(1.549, value(m.o), delta=1e-3)

    def test_gurobi_minlp_sqrt(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(0, 2))
        m.o = Objective(expr=sqrt(m.x) - (m.x**2) / 3, sense=maximize)
        gurobi_direct.solve(m)
        self.assertAlmostEqual(0.825, value(m.x), delta=1e-3)
        self.assertAlmostEqual(0.681, value(m.o), delta=1e-3)

    def test_gurobi_minlp_log(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(1, 2))
        m.o = Objective(expr=(m.x * m.x) / log(m.x))
        gurobi_direct.solve(m)
        self.assertAlmostEqual(sqrt(math.e), value(m.x), delta=1e-3)
        self.assertAlmostEqual(2 * math.e, value(m.o), delta=1e-3)

    def test_gurobi_minlp_log10(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(1, 2))
        m.o = Objective(expr=(m.x * m.x) / log10(m.x))
        gurobi_direct.solve(m)
        self.assertAlmostEqual(1.649, value(m.x), delta=1e-3)
        self.assertAlmostEqual(12.518, value(m.o), delta=1e-3)

    def test_gurobi_minlp_sigmoid(self):
        m = ConcreteModel(name="test")
        m.x = Var(bounds=(0, 4))
        m.y = Var(bounds=(0, 4))
        m.o = Objective(expr=m.y - m.x)
        m.c = Constraint(expr=1 / (1 + exp(-m.x)) <= m.y)
        gurobi_direct.solve(m)
        self.assertAlmostEqual(value(m.o), -3.017, delta=1e-3)

    def _build_divpwr_model(self, divide: bool, min: bool):
        model = ConcreteModel(name="test")
        model.x1 = Var(domain=NonNegativeReals, bounds=(0.5, 0.6))
        model.x2 = Var(domain=NonNegativeReals, bounds=(0.1, 0.2))
        model.y = Var(domain=Binary, initialize=1)

        y_mult = 1.3
        if divide:
            obj = (1 - model.y) / model.x1 + model.y * y_mult / model.x2
        else:
            obj = (1 - model.y) * (model.x1**-1) + model.y * y_mult * (model.x2**-1)

        if min:
            model.OBJ = Objective(expr=-1 * obj, sense=minimize)
        else:
            model.OBJ = Objective(expr=obj, sense=maximize)

        return model

    def test_gurobi_minlp_divpwr(self):
        params = [
            {"min": False, "divide": False, "obj": 13},
            {"min": False, "divide": True, "obj": 13},
            {"min": True, "divide": False, "obj": -13},
            {"min": True, "divide": True, "obj": -13},
        ]
        for p in params:
            model = self._build_divpwr_model(p['divide'], p['min'])
            gurobi_direct.solve(model)
            self.assertEqual(p["obj"], value(model.OBJ))
            self.assertEqual(1, model.y.value)

    def test_gurobi_minlp_acopf(self):
        # Based on https://docs.gurobi.com/projects/examples/en/current/examples/python/acopf_4buses.html

        # Number of Buses (Nodes)
        N = 4

        # Conductance/susceptance components
        G = [
            [1.7647, -0.5882, 0.0, -1.1765],
            [-0.5882, 1.5611, -0.3846, -0.5882],
            [0.0, -0.3846, 1.5611, -1.1765],
            [-1.1765, -0.5882, -1.1765, 2.9412],
        ]
        B = [
            [-7.0588, 2.3529, 0.0, 4.7059],
            [2.3529, -6.629, 1.9231, 2.3529],
            [0.0, 1.9231, -6.629, 4.7059],
            [4.7059, 2.3529, 4.7059, -11.7647],
        ]

        # Assign bounds where fixings are needed
        v_lb = [1.0, 0.0, 1.0, 0.0]
        v_ub = [1.0, 1.5, 1.0, 1.5]
        P_lb = [-3.0, -0.3, 0.3, -0.2]
        P_ub = [3.0, -0.3, 0.3, -0.2]
        Q_lb = [-3.0, -0.2, -3.0, -0.15]
        Q_ub = [3.0, -0.2, 3.0, -0.15]
        theta_lb = [0.0, -math.pi / 2, -math.pi / 2, -math.pi / 2]
        theta_ub = [0.0, math.pi / 2, math.pi / 2, math.pi / 2]

        exp_v = [1.0, 0.949, 1.0, 0.973]
        exp_theta = [0.0, -2.176, 1.046, -0.768]
        exp_P = [0.2083, -0.3, 0.3, -0.2]
        exp_Q = [0.212, -0.2, 0.173, -0.15]

        m = ConcreteModel(name="acopf")

        m.P = VarList()
        m.Q = VarList()
        m.v = VarList()
        m.theta = VarList()

        for i in range(N):
            p = m.P.add()
            p.lb = P_lb[i]
            p.ub = P_ub[i]

            q = m.Q.add()
            q.lb = Q_lb[i]
            q.ub = Q_ub[i]

            v = m.v.add()
            v.lb = v_lb[i]
            v.ub = v_ub[i]

            theta = m.theta.add()
            theta.lb = theta_lb[i]
            theta.ub = theta_ub[i]

        m.obj = Objective(expr=m.Q[1] + m.Q[3])

        m.define_P = ConstraintList()
        m.define_Q = ConstraintList()
        for i in range(N):
            m.define_P.add(
                m.P[i + 1]
                == m.v[i + 1]
                * sum(
                    m.v[j + 1]
                    * (
                        G[i][j] * cos(m.theta[i + 1] - m.theta[j + 1])
                        + B[i][j] * sin(m.theta[i + 1] - m.theta[j + 1])
                    )
                    for j in range(N)
                )
            )
            m.define_Q.add(
                m.Q[i + 1]
                == m.v[i + 1]
                * sum(
                    m.v[j + 1]
                    * (
                        G[i][j] * sin(m.theta[i + 1] - m.theta[j + 1])
                        - B[i][j] * cos(m.theta[i + 1] - m.theta[j + 1])
                    )
                    for j in range(N)
                )
            )

        results = gurobi_direct.solve(m, tee=True)
        self.assertEqual(
            results.termination_condition,
            TerminationCondition.convergenceCriteriaSatisfied,
        )
        self.assertEqual(results.solution_status, SolutionStatus.optimal)
        for i in range(N):
            self.assertAlmostEqual(
                exp_P[i], m.P[i + 1].value, delta=1e-3, msg=f'P[{i}]'
            )
            self.assertAlmostEqual(
                exp_Q[i], m.Q[i + 1].value, delta=1e-3, msg=f'Q[{i}]'
            )
            self.assertAlmostEqual(
                exp_v[i], m.v[i + 1].value, delta=1e-3, msg=f'v[{i}]'
            )
            self.assertAlmostEqual(
                exp_theta[i],
                m.theta[i + 1].value * 180 / math.pi,
                delta=1e-3,
                msg=f'theta[{i}]',
            )
