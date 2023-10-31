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

from pyomo.environ import ConcreteModel, Var, Objective, ConstraintList, SolverFactory

gurobi_lp = SolverFactory('gurobi', solver_io='lp')
gurobi_nl = SolverFactory('gurobi', solver_io='nl')
gurobi_direct = SolverFactory('gurobi_direct')
gurobi_persistent = SolverFactory('gurobi_persistent')
gurobi_appsi = SolverFactory('appsi_gurobi')

cplex_lp = SolverFactory('cplex', solver_io='lp')
cplex_nl = SolverFactory('cplex', solver_io='nl')
cplex_direct = SolverFactory('cplex_direct')
cplex_persistent = SolverFactory('cplex_persistent')
cplex_appsi = SolverFactory('appsi_cplex')

xpress_lp = SolverFactory('xpress', solver_io='lp')
xpress_nl = SolverFactory('xpress', solver_io='nl')
xpress_direct = SolverFactory('xpress_direct')
xpress_persistent = SolverFactory('xpress_persistent')
xpress_appsi = SolverFactory('appsi_xpress')


class TestQuadraticModels(unittest.TestCase):
    def _qp_model(self):
        m = ConcreteModel(name="test")
        m.x = Var([0, 1, 2])
        m.obj = Objective(
            expr=m.x[0]
            + 10 * m.x[1]
            + 100 * m.x[2]
            + 1000 * m.x[1] * m.x[2]
            + 10000 * m.x[0] ** 2
            + 10000 * m.x[1] ** 2
            + 100000 * m.x[2] ** 2
        )
        m.c = ConstraintList()
        m.c.add(m.x[0] == 1)
        m.c.add(m.x[1] == 2)
        m.c.add(m.x[2] == 4)
        return m

    @unittest.skipUnless(
        gurobi_lp.available(exception_flag=False), "needs Gurobi LP interface"
    )
    def test_qp_objective_gurobi_lp(self):
        m = self._qp_model()
        results = gurobi_lp.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        gurobi_nl.available(exception_flag=False), "needs Gurobi NL interface"
    )
    def test_qp_objective_gurobi_nl(self):
        m = self._qp_model()
        results = gurobi_nl.solve(m)
        # TODO: the NL interface should set either the Upper or Lower
        # bound for feasible solutions!
        #
        # self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])
        self.assertIn(str(int(m.obj())), results['Solver'][0]['Message'])

    @unittest.skipUnless(
        gurobi_appsi.available(exception_flag=False), "needs Gurobi APPSI interface"
    )
    def test_qp_objective_gurobi_appsi(self):
        m = self._qp_model()
        gurobi_appsi.set_instance(m)
        results = gurobi_appsi.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        gurobi_direct.available(exception_flag=False), "needs Gurobi Direct interface"
    )
    def test_qp_objective_gurobi_direct(self):
        m = self._qp_model()
        results = gurobi_direct.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        gurobi_persistent.available(exception_flag=False),
        "needs Gurobi Persistent interface",
    )
    def test_qp_objective_gurobi_persistent(self):
        m = self._qp_model()
        gurobi_persistent.set_instance(m)
        results = gurobi_persistent.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        cplex_lp.available(exception_flag=False), "needs Cplex LP interface"
    )
    def test_qp_objective_cplex_lp(self):
        m = self._qp_model()
        results = cplex_lp.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        cplex_nl.available(exception_flag=False), "needs Cplex NL interface"
    )
    def test_qp_objective_cplex_nl(self):
        m = self._qp_model()
        results = cplex_nl.solve(m)
        # TODO: the NL interface should set either the Upper or Lower
        # bound for feasible solutions!
        #
        # self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])
        self.assertIn(str(int(m.obj())), results['Solver'][0]['Message'])

    @unittest.skipUnless(
        cplex_direct.available(exception_flag=False), "needs Cplex Direct interface"
    )
    def test_qp_objective_cplex_direct(self):
        m = self._qp_model()
        results = cplex_direct.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        cplex_persistent.available(exception_flag=False),
        "needs Cplex Persistent interface",
    )
    def test_qp_objective_cplex_persistent(self):
        m = self._qp_model()
        cplex_persistent.set_instance(m)
        results = cplex_persistent.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        cplex_appsi.available(exception_flag=False), "needs Cplex APPSI interface"
    )
    def test_qp_objective_cplex_appsi(self):
        m = self._qp_model()
        cplex_appsi.set_instance(m)
        results = cplex_appsi.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        xpress_lp.available(exception_flag=False), "needs Xpress LP interface"
    )
    def test_qp_objective_xpress_lp(self):
        m = self._qp_model()
        results = xpress_lp.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        xpress_nl.available(exception_flag=False), "needs Xpress NL interface"
    )
    def test_qp_objective_xpress_nl(self):
        m = self._qp_model()
        results = xpress_nl.solve(m)
        # TODO: the NL interface should set either the Upper or Lower
        # bound for feasible solutions!
        #
        # self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])
        self.assertIn(str(int(m.obj())), results['Solver'][0]['Message'])

    @unittest.skipUnless(
        xpress_direct.available(exception_flag=False), "needs Xpress Direct interface"
    )
    def test_qp_objective_xpress_direct(self):
        m = self._qp_model()
        results = xpress_direct.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        xpress_persistent.available(exception_flag=False),
        "needs Xpress Persistent interface",
    )
    def test_qp_objective_xpress_persistent(self):
        m = self._qp_model()
        xpress_persistent.set_instance(m)
        results = xpress_persistent.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])

    @unittest.skipUnless(
        xpress_appsi.available(exception_flag=False), "needs Xpress APPSI interface"
    )
    def test_qp_objective_xpress_appsi(self):
        m = self._qp_model()
        xpress_appsi.set_instance(m)
        results = xpress_appsi.solve(m)
        self.assertEqual(m.obj(), results['Problem'][0]['Upper bound'])
