#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#

import pyomo.common.unittest as unittest

import pyomo.environ as pyo

from pyomo.common.dependencies import numpy as np, scipy_available, numpy_available
from pyomo.common.log import LoggingIntercept
from pyomo.repn.plugins.standard_form import LinearStandardFormCompiler

for sol in ['glpk', 'cbc', 'gurobi', 'cplex', 'xpress']:
    linear_solver = pyo.SolverFactory(sol)
    if linear_solver.available(exception_flag=False):
        break
else:
    linear_solver = None


@unittest.skipUnless(
    scipy_available & numpy_available, "standard_form requires scipy and numpy"
)
class TestLinearStandardFormCompiler(unittest.TestCase):
    def test_linear_model(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        repn = LinearStandardFormCompiler().write(m)

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        self.assertTrue(np.all(repn.A == np.array([[-1, -2, 0], [0, 1, 4]])))
        self.assertTrue(np.all(repn.rhs == np.array([-3, 5])))

    def test_linear_model_row_col_order(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        repn = LinearStandardFormCompiler().write(
            m, column_order=[m.y[3], m.y[2], m.x, m.y[1]], row_order=[m.d, m.c]
        )

        self.assertTrue(np.all(repn.c == np.array([0, 0, 0])))
        self.assertTrue(np.all(repn.A == np.array([[4, 0, 1], [0, -1, -2]])))
        self.assertTrue(np.all(repn.rhs == np.array([5, -3])))

    def test_suffix_warning(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([1, 2, 3])
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)

        m.dual = pyo.Suffix(direction=pyo.Suffix.EXPORT)
        m.b = pyo.Block()
        m.b.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

        with LoggingIntercept() as LOG:
            repn = LinearStandardFormCompiler().write(m)
        self.assertEqual(LOG.getvalue(), "")

        m.dual[m.c] = 5
        with LoggingIntercept() as LOG:
            repn = LinearStandardFormCompiler().write(m)
        self.assertEqual(
            LOG.getvalue(),
            "EXPORT Suffix 'dual' found on 1 block:\n"
            "    dual\n"
            "Standard Form compiler ignores export suffixes.  Skipping.\n",
        )

        m.b.dual[m.d] = 1
        with LoggingIntercept() as LOG:
            repn = LinearStandardFormCompiler().write(m)
        self.assertEqual(
            LOG.getvalue(),
            "EXPORT Suffix 'dual' found on 2 blocks:\n"
            "    dual\n"
            "    b.dual\n"
            "Standard Form compiler ignores export suffixes.  Skipping.\n",
        )

    def _verify_solution(self, soln, repn, eq):
        # clear out any old solution
        for v, val in soln:
            v.value = None
        for v in repn.x:
            v.value = None

        x = np.array(repn.x, dtype=object)
        ax = repn.A.todense() @ x

        def c_rule(m, i):
            if eq:
                return ax[i] == repn.b[i]
            else:
                return ax[i] <= repn.b[i]

        test_model = pyo.ConcreteModel()
        test_model.o = pyo.Objective(expr=repn.c[[1], :].todense()[0] @ x)
        test_model.c = pyo.Constraint(range(len(repn.b)), rule=c_rule)
        linear_solver.solve(test_model, tee=True)

        # Propagate any solution back to the original variables
        for v, expr in repn.eliminated_vars:
            v.value = pyo.value(expr)
        self.assertEqual(*zip(*((v.value, val) for v, val in soln)))

    @unittest.skipIf(
        linear_solver is None, 'verifying results requires a linear solver'
    )
    def test_alternative_forms(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var(
            [0, 1, 3], bounds=lambda m, i: (-1 * (i % 2) * 5, 10 - 12 * (i // 2))
        )
        m.c = pyo.Constraint(expr=m.x + 2 * m.y[1] >= 3)
        m.d = pyo.Constraint(expr=m.y[1] + 4 * m.y[3] <= 5)
        m.e = pyo.Constraint(expr=pyo.inequality(-2, m.y[0] + 1 + 6 * m.y[1], 7))
        m.f = pyo.Constraint(expr=m.x + m.y[0] + 2 == 10)
        m.o = pyo.Objective([1, 3], rule=lambda m, i: m.x + i * 5 * m.y[i])
        m.o[1].sense = pyo.maximize

        col_order = [m.x, m.y[0], m.y[1], m.y[3]]

        m.o[1].deactivate()
        linear_solver.solve(m)
        m.o[1].activate()
        soln = [(v, v.value) for v in col_order]

        repn = LinearStandardFormCompiler().write(m, column_order=col_order)

        self.assertEqual(
            repn.rows, [(m.c, -1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 1), (m.f, -1)]
        )
        self.assertEqual(repn.x, [m.x, m.y[0], m.y[1], m.y[3]])
        ref = np.array(
            [
                [-1, 0, -2, 0],
                [0, 0, 1, 4],
                [0, 1, 6, 0],
                [0, -1, -6, 0],
                [1, 1, 0, 0],
                [-1, -1, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([-3, 5, 6, 3, 8, -8])))
        self.assertTrue(np.all(repn.c == np.array([[-1, 0, -5, 0], [1, 0, 0, 15]])))
        self._verify_solution(soln, repn, False)

        repn = LinearStandardFormCompiler().write(
            m, nonnegative_vars=True, column_order=col_order
        )

        self.assertEqual(
            repn.rows, [(m.c, -1), (m.d, 1), (m.e, 1), (m.e, -1), (m.f, 1), (m.f, -1)]
        )
        self.assertEqual(
            list(map(str, repn.x)),
            ['_neg_0', '_pos_0', 'y[0]', '_neg_2', '_pos_2', '_neg_3'],
        )
        ref = np.array(
            [
                [1, -1, 0, 2, -2, 0],
                [0, 0, 0, -1, 1, -4],
                [0, 0, 1, -6, 6, 0],
                [0, 0, -1, 6, -6, 0],
                [-1, 1, 1, 0, 0, 0],
                [1, -1, -1, 0, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([-3, 5, 6, 3, 8, -8])))
        self.assertTrue(
            np.all(repn.c == np.array([[1, -1, 0, 5, -5, 0], [-1, 1, 0, 0, 0, -15]]))
        )
        self._verify_solution(soln, repn, False)

        repn = LinearStandardFormCompiler().write(
            m, slack_form=True, column_order=col_order
        )

        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.f, 1)])
        self.assertEqual(
            list(map(str, repn.x)),
            ['x', 'y[0]', 'y[1]', 'y[3]', '_slack_0', '_slack_1', '_slack_2'],
        )
        self.assertEqual(
            list(v.bounds for v in repn.x),
            [(None, None), (0, 10), (-5, 10), (-5, -2), (None, 0), (0, None), (-9, 0)],
        )
        ref = np.array(
            [
                [1, 0, 2, 0, 1, 0, 0],
                [0, 0, 1, 4, 0, 1, 0],
                [0, 1, 6, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([3, 5, -3, 8])))
        self.assertTrue(
            np.all(
                repn.c == np.array([[-1, 0, -5, 0, 0, 0, 0], [1, 0, 0, 15, 0, 0, 0]])
            )
        )
        self._verify_solution(soln, repn, True)

        repn = LinearStandardFormCompiler().write(
            m, slack_form=True, nonnegative_vars=True, column_order=col_order
        )

        self.assertEqual(repn.rows, [(m.c, 1), (m.d, 1), (m.e, 1), (m.f, 1)])
        self.assertEqual(
            list(map(str, repn.x)),
            [
                '_neg_0',
                '_pos_0',
                'y[0]',
                '_neg_2',
                '_pos_2',
                '_neg_3',
                '_neg_4',
                '_slack_1',
                '_neg_6',
            ],
        )
        self.assertEqual(
            list(v.bounds for v in repn.x),
            [
                (0, None),
                (0, None),
                (0, 10),
                (0, 5),
                (0, 10),
                (2, 5),
                (0, None),
                (0, None),
                (0, 9),
            ],
        )
        ref = np.array(
            [
                [-1, 1, 0, -2, 2, 0, -1, 0, 0],
                [0, 0, 0, -1, 1, -4, 0, 1, 0],
                [0, 0, 1, -6, 6, 0, 0, 0, -1],
                [-1, 1, 1, 0, 0, 0, 0, 0, 0],
            ]
        )
        self.assertTrue(np.all(repn.A == ref))
        self.assertTrue(np.all(repn.b == np.array([3, 5, -3, 8])))
        ref = np.array([[1, -1, 0, 5, -5, 0, 0, 0, 0], [-1, 1, 0, 0, 0, -15, 0, 0, 0]])
        self.assertTrue(np.all(repn.c == ref))
        self._verify_solution(soln, repn, True)
