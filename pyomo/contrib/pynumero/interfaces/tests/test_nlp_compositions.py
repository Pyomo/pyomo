#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import pyutilib.th as unittest
import pyomo.environ as aml
import os

from .. import numpy_available, scipy_available
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import scipy.sparse as spa
import numpy as np

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")


from pyomo.contrib.pynumero.interfaces.nlp import PyomoNLP, NLP
from pyomo.contrib.pynumero.interfaces.nlp_compositions import TwoStageStochasticNLP
from pyomo.contrib.pynumero.sparse import (BlockVector,
                                           BlockMatrix,
                                           BlockSymMatrix,
                                           empty_matrix)

from scipy.sparse import coo_matrix, identity


def create_basic_dense_qp(G, A, b, c, complicated_var_ids):

    nx = G.shape[0]
    nl = A.shape[0]

    model = aml.ConcreteModel()
    model.var_ids = range(nx)
    model.complicated_var_ids = complicated_var_ids
    model.con_ids = range(nl)

    model.x = aml.Var(model.var_ids, initialize=0.0)

    model.x[0].value = 1.0
    model.x[0].setlb(-100.0)
    model.x[0].setub(100.0)

    model.z = aml.Var(model.complicated_var_ids, initialize=0.0)
    model.hessian_f = aml.Param(model.var_ids, model.var_ids, mutable=True, rule=lambda m, i, j: G[i, j])
    model.jacobian_c = aml.Param(model.con_ids, model.var_ids, mutable=True, rule=lambda m, i, j: A[i, j])
    model.rhs = aml.Param(model.con_ids, mutable=True, rule=lambda m, i: b[i])
    model.grad_f = aml.Param(model.var_ids, mutable=True, rule=lambda m, i: c[i])

    def equality_constraint_rule(m, i):
        return sum(m.jacobian_c[i, j] * m.x[j] for j in m.var_ids) == m.rhs[i]
    model.equalities = aml.Constraint(model.con_ids, rule=equality_constraint_rule)

    def fixing_constraints_rule(m, i):
        return m.z[i] == m.x[i]
    model.fixing_constraints = aml.Constraint(model.complicated_var_ids, rule=fixing_constraints_rule)

    def second_stage_cost_rule(m):
        accum = 0.0
        for i in m.var_ids:
            accum += m.x[i] * sum(m.hessian_f[i, j] * m.x[j] for j in m.var_ids)
        accum *= 0.5
        accum += sum(m.x[j] * m.grad_f[j] for j in m.var_ids)
        return accum

    model.FirstStageCost = aml.Expression(expr=0.0)
    model.SecondStageCost = aml.Expression(rule=second_stage_cost_rule)

    model.obj = aml.Objective(expr=model.FirstStageCost + model.SecondStageCost,
                             sense=aml.minimize)

    return model


def create_basic_model():

    m = aml.ConcreteModel()
    m.x = aml.Var([1, 2, 3], domain=aml.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.c1 = aml.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.c2 = aml.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.d1 = aml.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.d2 = aml.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.d3 = aml.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = aml.Objective(expr=m.x[2]**2)
    return m


@unittest.skipIf(os.name in ['nt', 'dos'], "Do not test on windows")
class TestTwoStageStochasticNLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        # Hessian
        cls.G = np.array([[36, 17, 19, 12, 8, 15],
                          [17, 33, 18, 11, 7, 14],
                          [19, 18, 43, 13, 8, 16],
                          [12, 11, 13, 18, 6, 11],
                          [8, 7, 8, 6, 9, 8],
                          [15, 14, 16, 11, 8, 29]])

        # jacobian
        cls.A = np.array([[7, 1, 8, 3, 3, 3],
                          [5, 0, 5, 1, 5, 8],
                          [2, 6, 7, 1, 1, 8],
                          [1, 5, 0, 6, 1, 0]])

        cls.b = np.array([84, 62, 65, 1])
        cls.c = np.array([20, 15, 21, 18, 29, 24])

        cls.complicated_vars_ids = [4, 5]

        cls.scenarios = dict()
        cls.coupling_vars = dict()
        cls.n_scenarios = 3
        for i in range(cls.n_scenarios):
            instance = create_basic_dense_qp(cls.G,
                                             cls.A,
                                             cls.b,
                                             cls.c,
                                             cls.complicated_vars_ids)

            nlp = PyomoNLP(instance)
            scenario_name = "s{}".format(i)
            cls.scenarios[scenario_name] = nlp

            cvars = list()
            for k in cls.complicated_vars_ids:
                cvars.append(nlp.variable_idx(instance.z[k]))
            cls.coupling_vars[scenario_name] = cvars

        cls.nlp = TwoStageStochasticNLP(cls.scenarios, cls.coupling_vars)

        # create second stochastic nlp for testing
        cls.complicated_vars_ids2 = [1, 3]
        cls.scenarios2 = dict()
        cls.coupling_vars2 = dict()
        cls.n_scenarios2 = 3
        for i in range(cls.n_scenarios2):
            instance = create_basic_model()
            nlp = PyomoNLP(instance)
            scenario_name = "s{}".format(i)
            cls.scenarios2[scenario_name] = nlp
            cvars = []
            for k in cls.complicated_vars_ids2:
                cvars.append(nlp.variable_idx(instance.x[k]))
            cls.coupling_vars2[scenario_name] = cvars

        cls.nlp2 = TwoStageStochasticNLP(cls.scenarios2, cls.coupling_vars2)

    def test_nx(self):
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        nx = self.n_scenarios * nx_i + nz
        self.assertEqual(self.nlp.nx, nx)

        nz = len(self.complicated_vars_ids2)
        nx_i = self.scenarios2['s0'].nx
        nx = self.n_scenarios2 * nx_i + nz
        self.assertEqual(self.nlp2.nx, nx)

    def test_ng(self):
        nz = len(self.complicated_vars_ids)
        ng_i = self.A.shape[0] + nz
        ng_z = nz * self.n_scenarios
        ng = ng_i * self.n_scenarios + ng_z
        self.assertEqual(self.nlp.ng, ng)

        nz = len(self.complicated_vars_ids2)
        ng_i = self.scenarios2['s0'].ng
        ng_z = nz * self.n_scenarios2
        ng = ng_i * self.n_scenarios2 + ng_z
        self.assertEqual(self.nlp2.ng, ng)

    def test_nc(self):

        nz = len(self.complicated_vars_ids2)
        nc_i = self.scenarios2['s0'].nc
        nc_z = nz * self.n_scenarios2
        nc = nc_i * self.n_scenarios2 + nc_z
        self.assertEqual(self.nlp2.nc, nc)

    def test_nd(self):
        nd_i = self.scenarios2['s0'].nd
        nd = nd_i * self.n_scenarios2
        self.assertEqual(self.nlp2.nd, nd)

    def test_x_maps(self):

        lower_mask = np.array([False, True, True], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        lower_mask_z = np.array([False for i in range(nz)], dtype=bool)
        composite_lower_mask = self.nlp2._lower_x_mask
        self.assertIsInstance(composite_lower_mask, BlockVector)
        n_scenarios = len(self.scenarios2)
        self.assertEqual(composite_lower_mask.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_lower_mask[i], lower_mask))
        self.assertTrue(np.allclose(composite_lower_mask[n_scenarios], lower_mask_z))

        upper_mask = np.array([False, True, False], dtype=bool)
        upper_mask_z = np.array([False for i in range(nz)], dtype=bool)
        composite_upper_mask = self.nlp2._upper_x_mask
        self.assertIsInstance(composite_upper_mask, BlockVector)
        self.assertEqual(composite_upper_mask.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_upper_mask[i], upper_mask))
        self.assertTrue(np.allclose(composite_lower_mask[n_scenarios], upper_mask_z))

        lower_map = np.array([1, 2])
        lower_map_z = np.zeros(0)
        composite_lower_map = self.nlp2._lower_x_map
        self.assertIsInstance(composite_lower_map, BlockVector)
        self.assertEqual(composite_lower_map.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_lower_map[i], lower_map))
        self.assertTrue(np.allclose(composite_lower_map[n_scenarios], lower_map_z))

        upper_map = np.array([1])
        upper_map_z = np.zeros(0)
        composite_upper_map = self.nlp2._upper_x_map
        self.assertIsInstance(composite_upper_map, BlockVector)
        self.assertEqual(composite_upper_map.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_upper_map[i], upper_map))
        self.assertTrue(np.allclose(composite_upper_map[n_scenarios], upper_map_z))

    def test_g_maps(self):

        n_scenarios = len(self.scenarios2)
        lower_mask = np.array([True, True, False, True, True], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        lower_mask_z = np.array([True for i in range(nz)], dtype=bool)
        composite_lower_mask = self.nlp2._lower_g_mask
        self.assertIsInstance(composite_lower_mask, BlockVector)
        self.assertEqual(composite_lower_mask.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_lower_mask[i], lower_mask))
            self.assertTrue(np.allclose(composite_lower_mask[n_scenarios+i],
                                        lower_mask_z))

        n_scenarios = len(self.scenarios2)
        upper_mask = np.array([True, True, True, False, False], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        upper_mask_z = np.array([True for i in range(nz)], dtype=bool)
        composite_upper_mask = self.nlp2._upper_g_mask
        self.assertIsInstance(composite_upper_mask, BlockVector)
        self.assertEqual(composite_upper_mask.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_upper_mask[i], upper_mask))
            self.assertTrue(np.allclose(composite_upper_mask[n_scenarios + i],
                                        upper_mask_z))

        lower_map = np.array([0, 1, 3, 4])
        instance = self.scenarios2['s0']
        lower_map_z = np.arange(nz) #+ instance.ng
        composite_lower_map = self.nlp2._lower_g_map
        self.assertIsInstance(composite_lower_map, BlockVector)
        self.assertEqual(composite_lower_map.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_lower_map[i], lower_map))
            self.assertTrue(np.allclose(composite_lower_map[i + n_scenarios], lower_map_z))

        upper_map = np.array([0, 1, 2])
        instance = self.scenarios2['s0']
        upper_map_z = np.arange(nz)  # + instance.ng
        composite_upper_map = self.nlp2._upper_g_map
        self.assertIsInstance(composite_upper_map, BlockVector)
        self.assertEqual(composite_upper_map.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_upper_map[i], upper_map))
            self.assertTrue(np.allclose(composite_upper_map[i + n_scenarios], upper_map_z))

    def test_c_maps(self):

        n_scenarios = len(self.scenarios2)
        c_mask = np.array([True, True, False, False, False], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        c_mask_z = np.array([True for i in range(nz)], dtype=bool)
        composite_c_mask = self.nlp2._c_mask
        self.assertIsInstance(composite_c_mask, BlockVector)
        self.assertEqual(composite_c_mask.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_c_mask[i], c_mask))
            self.assertTrue(np.allclose(composite_c_mask[n_scenarios + i],
                                        c_mask_z))

        c_map = np.array([0, 1])
        instance = self.scenarios2['s0']
        c_map_z = np.arange(nz)  # + instance.ng
        composite_c_map = self.nlp2._c_map
        self.assertIsInstance(composite_c_map, BlockVector)
        self.assertEqual(composite_c_map.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_c_map[i], c_map))
            self.assertTrue(np.allclose(composite_c_map[i + n_scenarios], c_map_z))

    def test_d_maps(self):

        n_scenarios = len(self.scenarios2)
        d_mask = np.array([False, False, True, True, True], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        d_mask_z = np.zeros(nz, dtype=bool)
        composite_d_mask = self.nlp2._d_mask
        self.assertIsInstance(composite_d_mask, BlockVector)
        self.assertEqual(composite_d_mask.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_d_mask[i], d_mask))
            self.assertTrue(np.allclose(composite_d_mask[n_scenarios + i],
                                        d_mask_z))

        d_map = np.array([2, 3, 4])
        instance = self.scenarios2['s0']
        d_map_z = np.arange(0)  # + instance.ng
        composite_d_map = self.nlp2._d_map
        self.assertIsInstance(composite_d_map, BlockVector)
        self.assertEqual(composite_d_map.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_d_map[i], d_map))
            self.assertTrue(np.allclose(composite_d_map[i + n_scenarios], d_map_z))

        n_scenarios = len(self.scenarios2)
        lower_mask = np.array([False, True, True], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        composite_lower_mask = self.nlp2._lower_d_mask
        self.assertIsInstance(composite_lower_mask, BlockVector)
        self.assertEqual(composite_lower_mask.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_lower_mask[i], lower_mask))

        n_scenarios = len(self.scenarios2)
        upper_mask = np.array([True, False, False], dtype=bool)
        nz = len(self.complicated_vars_ids2)
        composite_upper_mask = self.nlp2._upper_d_mask
        self.assertIsInstance(composite_upper_mask, BlockVector)
        self.assertEqual(composite_upper_mask.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_upper_mask[i], upper_mask))

        lower_map = np.array([1, 2])
        instance = self.scenarios2['s0']
        composite_lower_map = self.nlp2._lower_d_map
        self.assertIsInstance(composite_lower_map, BlockVector)
        self.assertEqual(composite_lower_map.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_lower_map[i], lower_map))

        upper_map = np.array([0])
        instance = self.scenarios2['s0']
        composite_upper_map = self.nlp2._upper_d_map
        self.assertIsInstance(composite_upper_map, BlockVector)
        self.assertEqual(composite_upper_map.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(composite_upper_map[i], upper_map))

    def test_nblocks(self):
        self.assertEqual(self.n_scenarios, self.nlp.nblocks)
        self.assertEqual(self.n_scenarios2, self.nlp2.nblocks)

    def test_nz(self):
        nz = len(self.complicated_vars_ids)
        self.assertEqual(self.nlp.nz, nz)

        nz = len(self.complicated_vars_ids2)
        self.assertEqual(self.nlp2.nz, nz)

    def test_xl(self):
        xl = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            xl[i] = np.array([-np.inf]*nx_i)
            xl[i][0] = -100.0
        xl[self.n_scenarios] = np.array([-np.inf] * nz)
        self.assertIsInstance(self.nlp.xl(), BlockVector)
        xl_flat = xl.flatten()
        self.assertEqual(xl.nblocks, self.nlp.xl().nblocks)
        self.assertTrue(np.allclose(xl.block_sizes(), self.nlp.xl().block_sizes()))
        self.assertListEqual(list(xl_flat), list(self.nlp.xl().flatten()))

        # test second nlp
        xl = np.array([-np.inf, 0, 0])
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        xl_z = np.array([-np.inf for i in range(nz)])
        lower_x = self.nlp2.xl()
        self.assertIsInstance(lower_x, BlockVector)
        self.assertEqual(lower_x.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(lower_x[i], xl))
        self.assertTrue(np.allclose(lower_x[n_scenarios], xl_z))

        xl = np.array([0, 0])
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        xl_z = np.zeros(0)
        lower_x = self.nlp2.xl(condensed=True)
        self.assertIsInstance(lower_x, BlockVector)
        self.assertEqual(lower_x.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(lower_x[i], xl))
        self.assertTrue(np.allclose(lower_x[n_scenarios], xl_z))

    def test_xu(self):
        xu = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            xu[i] = np.array([np.inf]*nx_i)
            xu[i][0] = 100.0
        xu[self.n_scenarios] = np.array([np.inf] * nz)
        self.assertIsInstance(self.nlp.xu(), BlockVector)
        xu_flat = xu.flatten()
        self.assertEqual(xu.nblocks, self.nlp.xu().nblocks)
        self.assertTrue(np.allclose(xu.block_sizes(), self.nlp.xu().block_sizes()))
        self.assertListEqual(list(xu_flat), list(self.nlp.xu().flatten()))

        # test second nlp
        xu = np.array([np.inf, 100.0, np.inf])
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        xu_z = np.array([np.inf for i in range(nz)])
        upper_x = self.nlp2.xu()
        self.assertIsInstance(upper_x, BlockVector)
        self.assertEqual(upper_x.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(upper_x[i], xu))
        self.assertTrue(np.allclose(upper_x[n_scenarios], xu_z))

        xu = np.array([100.0])
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        xu_z = np.zeros(0)
        upper_x = self.nlp2.xu(condensed=True)
        self.assertIsInstance(upper_x, BlockVector)
        self.assertEqual(upper_x.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(upper_x[i], xu))
        self.assertTrue(np.allclose(upper_x[n_scenarios], xu_z))

    def test_gl(self):
        gl = [0.0, 0.0, -np.inf, -100., -500.]
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        lower_g = self.nlp2.gl()
        self.assertIsInstance(lower_g, BlockVector)
        self.assertEqual(lower_g.nblocks, n_scenarios * 2)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(lower_g[i], gl))
            self.assertTrue(np.allclose(lower_g[i+n_scenarios],
                                        np.zeros(nz)))

        gl = np.array([0.0, 0.0, -100., -500.])
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        gl_z = np.zeros(nz)
        lower_g = self.nlp2.gl(condensed=True)
        self.assertIsInstance(lower_g, BlockVector)
        self.assertEqual(lower_g.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(lower_g[i], gl))
            self.assertTrue(np.allclose(lower_g[i + n_scenarios],
                                        gl_z))

    def test_gu(self):
        gu = [0.0, 0.0, 100., np.inf, np.inf]
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        upper_g = self.nlp2.gu()
        self.assertIsInstance(upper_g, BlockVector)
        self.assertEqual(upper_g.nblocks, n_scenarios * 2)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(upper_g[i], gu))
            self.assertTrue(np.allclose(upper_g[i + n_scenarios],
                                        np.zeros(nz)))

        gu = np.array([0.0, 0.0, 100.])
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        gu_z = np.zeros(nz)
        upper_g = self.nlp2.gu(condensed=True)
        self.assertIsInstance(upper_g, BlockVector)
        self.assertEqual(upper_g.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(upper_g[i], gu))
            self.assertTrue(np.allclose(upper_g[i + n_scenarios],
                                        gu_z))

    def test_dl(self):
        dl = [-np.inf, -100., -500.]
        n_scenarios = len(self.scenarios2)
        lower_d = self.nlp2.dl()
        self.assertIsInstance(lower_d, BlockVector)
        self.assertEqual(lower_d.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(lower_d[i], dl))

        dl = np.array([-100., -500.])
        n_scenarios = len(self.scenarios2)
        lower_d = self.nlp2.dl(condensed=True)
        self.assertIsInstance(lower_d, BlockVector)
        self.assertEqual(lower_d.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(lower_d[i], dl))

    def test_du(self):
        du = [100., np.inf, np.inf]
        n_scenarios = len(self.scenarios2)
        upper_d = self.nlp2.du()
        self.assertIsInstance(upper_d, BlockVector)
        self.assertEqual(upper_d.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(upper_d[i], du))

        du = np.array([100.])
        n_scenarios = len(self.scenarios2)
        upper_d = self.nlp2.du(condensed=True)
        self.assertIsInstance(upper_d, BlockVector)
        self.assertEqual(upper_d.nblocks, n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(upper_d[i], du))

    def test_x_init(self):
        x_init = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            x_init[i] = np.zeros(nx_i)
            x_init[i][0] = 1.0
        x_init[self.n_scenarios] = np.zeros(nz)
        self.assertIsInstance(self.nlp.x_init(), BlockVector)
        x_init_flat = x_init.flatten()
        self.assertEqual(x_init.nblocks, self.nlp.x_init().nblocks)
        self.assertTrue(np.allclose(x_init.block_sizes(), self.nlp.x_init().block_sizes()))
        self.assertListEqual(list(x_init_flat), list(self.nlp.x_init().flatten()))

        # test second nlp
        x_init_i = np.array(range(1, 4))
        nz = len(self.complicated_vars_ids2)
        n_scenarios = len(self.scenarios2)
        x_init = self.nlp2.x_init()
        self.assertIsInstance(x_init, BlockVector)
        self.assertEqual(x_init.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(x_init[i], x_init_i))
        self.assertTrue(np.allclose(x_init[n_scenarios], np.zeros(nz)))

    def test_create_vector_x(self):

        x_ = BlockVector(self.n_scenarios + 1)
        nz = len(self.complicated_vars_ids)
        nx_i = (self.G.shape[0] + nz)
        for i in range(self.n_scenarios):
            x_[i] = np.zeros(nx_i)
        x_[self.n_scenarios] = np.zeros(nz)
        self.assertEqual(x_.shape, self.nlp.create_vector_x().shape)
        self.assertEqual(x_.nblocks,
                         self.nlp.create_vector_x().nblocks)
        self.assertTrue(np.allclose(x_.block_sizes(),
                                    self.nlp.create_vector_x().block_sizes()))
        self.assertListEqual(list(x_.flatten()),
                             list(self.nlp.create_vector_x().flatten()))

        # check for subset
        for s in ['l', 'u']:
            xs = self.nlp.create_vector_x(subset=s)
            xs_ = BlockVector(self.n_scenarios + 1)
            for i in range(self.n_scenarios):
                xs_[i] = np.zeros(1)
            xs_[self.n_scenarios] = np.zeros(0)
            self.assertEqual(xs_.shape, xs.shape)
            self.assertEqual(xs_.nblocks, xs.nblocks)
            self.assertTrue(np.allclose(xs_.block_sizes(),
                                        xs.block_sizes()))
            self.assertListEqual(list(xs_.flatten()),
                                 list(xs.flatten()))

        # test second nlp
        xi = np.zeros(3)
        nz = len(self.complicated_vars_ids2)
        n_scenarios = len(self.scenarios2)
        x = self.nlp2.create_vector_x()
        self.assertIsInstance(x, BlockVector)
        self.assertEqual(x.nblocks, n_scenarios + 1)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(x[i], xi))
        self.assertTrue(np.allclose(x[n_scenarios], np.zeros(nz)))

        for s in ['l', 'u']:
            if s == 'l':
                xi = np.zeros(2)
            else:
                xi = np.zeros(1)
            n_scenarios = len(self.scenarios2)
            x = self.nlp2.create_vector_x(subset=s)
            self.assertIsInstance(x, BlockVector)
            self.assertEqual(x.nblocks, n_scenarios + 1)
            for i in range(n_scenarios):
                self.assertTrue(np.allclose(x[i], xi))
            self.assertTrue(np.allclose(x[n_scenarios], np.zeros(0)))

    def test_create_vector_y(self):
        nz = len(self.complicated_vars_ids)
        ng_i = self.A.shape[0] + nz

        y_ = BlockVector(2 * self.n_scenarios)
        for i in range(self.n_scenarios):
            y_[i] = np.zeros(ng_i)
            y_[self.n_scenarios + i] = np.zeros(nz)
        y = self.nlp.create_vector_y()

        self.assertEqual(y_.shape, y.shape)
        self.assertEqual(y_.nblocks, y.nblocks)
        self.assertTrue(np.allclose(y_.block_sizes(),
                                    y.block_sizes()))
        self.assertListEqual(list(y_.flatten()),
                             list(y.flatten()))

        # check for equalities
        ys_ = BlockVector(2 * self.n_scenarios)
        for i in range(self.n_scenarios):
            ys_[i] = np.zeros(ng_i)
            ys_[self.n_scenarios + i] = np.zeros(nz)
        ys = self.nlp.create_vector_y(subset='c')
        self.assertEqual(ys_.shape, ys.shape)
        self.assertEqual(ys_.nblocks, ys.nblocks)
        self.assertTrue(np.allclose(ys_.block_sizes(),
                                    ys.block_sizes()))
        self.assertListEqual(list(ys_.flatten()),
                             list(ys.flatten()))

        # check for inequalities
        ys_ = BlockVector(self.n_scenarios)
        for i in range(self.n_scenarios):
            ys_[i] = np.zeros(0)
        ys = self.nlp.create_vector_y(subset='d')
        self.assertEqual(ys_.shape, ys.shape)
        self.assertEqual(ys_.nblocks, ys.nblocks)
        self.assertTrue(np.allclose(ys_.block_sizes(),
                                    ys.block_sizes()))
        self.assertListEqual(list(ys_.flatten()),
                             list(ys.flatten()))

        # test second nlp
        instance = self.scenarios2['s0']
        nz = len(self.complicated_vars_ids2)
        n_scenarios = len(self.scenarios2)
        gi = instance.ng
        y = self.nlp2.create_vector_y()
        self.assertIsInstance(y, BlockVector)
        self.assertEqual(y.nblocks, 2 * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(y[i], np.zeros(gi)))
            self.assertTrue(np.allclose(y[i + n_scenarios], np.zeros(nz)))

        for s in ['c', 'd']:
            y = self.nlp2.create_vector_y(subset=s)
            self.assertIsInstance(y, BlockVector)
            if s == 'c':
                gi = 2
                self.assertEqual(y.nblocks, 2 * n_scenarios)
            elif s == 'd':
                gi = 3
                self.assertEqual(y.nblocks, n_scenarios)
            elif s == 'dl':
                gi = 2
                self.assertEqual(y.nblocks, n_scenarios)
            elif s == 'du':
                gi = 1
                self.assertEqual(y.nblocks, n_scenarios)
            for i in range(n_scenarios):
                self.assertTrue(np.allclose(y[i], np.zeros(gi)))
                if s == 'c':
                    self.assertTrue(np.allclose(y[i + n_scenarios], np.zeros(nz)))

    def test_nlps(self):

        counter = 0
        for name, nlp in self.nlp.nlps():
            counter += 1
            self.assertIsInstance(nlp, NLP)
        self.assertEqual(counter, self.n_scenarios)

        counter = 0
        for name, nlp in self.nlp2.nlps():
            counter += 1
            self.assertIsInstance(nlp, NLP)
        self.assertEqual(counter, self.n_scenarios2)

    def test_objective(self):

        G = self.G
        c = self.c
        x_ = np.ones(G.shape[1])
        single_obj = 0.5 * x_.transpose().dot(G.dot(x_)) + x_.dot(c)
        obj_ = single_obj * self.n_scenarios
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        obj = self.nlp.objective(x)
        self.assertEqual(obj, obj_)
        obj = self.nlp.objective(x.flatten())
        self.assertEqual(obj, obj_)

        # test nlp2
        x = self.nlp2.create_vector_x()
        n_scenarios = len(self.scenarios2)
        for i in range(n_scenarios):
            x[i][1] = 5
        self.assertEqual(25.0 * n_scenarios, self.nlp2.objective(x))

    def test_grad_objective(self):

        G = self.G
        c = self.c
        nz = len(self.complicated_vars_ids)
        x_ = np.ones(G.shape[1])
        single_grad = G.dot(x_) + c
        single_grad = np.append(single_grad, np.zeros(nz))
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        grad_obj = self.nlp.grad_objective(x)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(grad_obj[i], single_grad))
        self.assertTrue(np.allclose(grad_obj[self.n_scenarios],
                                    np.zeros(nz)))

        grad_obj.fill(0.0)
        self.nlp.grad_objective(x, out=grad_obj)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(grad_obj[i], single_grad))
        self.assertTrue(np.allclose(grad_obj[self.n_scenarios],
                                    np.zeros(nz)))

        grad_obj = self.nlp.grad_objective(x.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(grad_obj[i], single_grad))
        self.assertTrue(np.allclose(grad_obj[self.n_scenarios],
                                    np.zeros(nz)))

        # test nlp2
        x = self.nlp2.create_vector_x()
        nz = len(self.complicated_vars_ids2)
        n_scenarios = len(self.scenarios2)
        for i in range(n_scenarios):
            x[i][1] = 1

        df = self.nlp2.grad_objective(x)
        self.assertIsInstance(df, BlockVector)
        self.assertEqual(df.nblocks, n_scenarios + 1)
        dfi = np.zeros(3)
        dfi[1] = 2
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(df[i], dfi))
        self.assertTrue(np.allclose(df[n_scenarios], np.zeros(nz)))

    def test_evaluate_g(self):

        nz = len(self.complicated_vars_ids)
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        gi = np.array([-59, -38, -40, 12, 0, 0])
        g = self.nlp.evaluate_g(x)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
            self.assertTrue(np.allclose(g[i+self.n_scenarios], np.zeros(nz)))

        g.fill(0.0)
        self.nlp.evaluate_g(x, out=g)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
            self.assertTrue(np.allclose(g[i + self.n_scenarios], np.zeros(nz)))

        g = self.nlp.evaluate_g(x.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
            self.assertTrue(np.allclose(g[i + self.n_scenarios], np.zeros(nz)))

        g.fill(0.0)
        self.nlp.evaluate_g(x.flatten(), out=g)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
            self.assertTrue(np.allclose(g[i + self.n_scenarios], np.zeros(nz)))

        # test nlp2
        instance = self.scenarios2['s0']
        n_scenarios = len(self.scenarios2)
        xi = instance.x_init()
        gi = instance.evaluate_g(xi)
        ngi = gi.size

        x = self.nlp2.x_init()
        nz = len(self.complicated_vars_ids2)
        g = self.nlp2.evaluate_g(x)
        self.assertIsInstance(g, BlockVector)
        self.assertEqual(g.nblocks, 2 * n_scenarios)
        self.assertEqual(g.size, n_scenarios * (ngi + nz))
        cvars = [0, 2]
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
            self.assertTrue(np.allclose(g[i + n_scenarios], x[i][cvars]))

        # test out
        g.fill(0.0)
        self.nlp2.evaluate_g(x, out=g)
        self.assertIsInstance(g, BlockVector)
        self.assertEqual(g.nblocks, 2 * n_scenarios)
        self.assertEqual(g.size, n_scenarios * (ngi + nz))
        cvars = [0, 2]
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(g[i], gi))
            self.assertTrue(np.allclose(g[i + n_scenarios], x[i][cvars]))

    def test_evaluate_c(self):

        nz = len(self.complicated_vars_ids)
        x = self.nlp.create_vector_x()
        x.fill(1.0)
        ci = np.array([-59, -38, -40, 12, 0, 0])
        c = self.nlp.evaluate_c(x)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i+self.n_scenarios], np.zeros(nz)))

        c.fill(0.0)
        self.nlp.evaluate_c(x, out=c)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + self.n_scenarios], np.zeros(nz)))

        c = self.nlp.evaluate_c(x.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + self.n_scenarios], np.zeros(nz)))

        c.fill(0.0)
        self.nlp.evaluate_c(x.flatten(), out=c)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + self.n_scenarios], np.zeros(nz)))

        # test nlp2
        instance = self.scenarios2['s0']
        n_scenarios = len(self.scenarios2)
        xi = instance.x_init()
        ci = instance.evaluate_c(xi)
        nci = ci.size

        x = self.nlp2.x_init()
        nz = len(self.complicated_vars_ids2)
        c = self.nlp2.evaluate_c(x)
        self.assertIsInstance(c, BlockVector)
        self.assertEqual(c.nblocks, 2 * n_scenarios)
        self.assertEqual(c.size, n_scenarios * (nci + nz))
        cvars = [0, 2]
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + n_scenarios], x[i][cvars]))

        # test out
        c.fill(0.0)
        self.nlp2.evaluate_c(x, out=c)
        self.assertIsInstance(c, BlockVector)
        self.assertEqual(c.nblocks, 2 * n_scenarios)
        self.assertEqual(c.size, n_scenarios * (nci + nz))
        cvars = [0, 2]
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + n_scenarios], x[i][cvars]))

        # tests evaluated_g
        g = self.nlp2.evaluate_g(x)
        c = self.nlp2.evaluate_c(x, evaluated_g=g)
        self.assertIsInstance(c, BlockVector)
        self.assertEqual(c.nblocks, 2 * n_scenarios)
        self.assertEqual(c.size, n_scenarios * (nci + nz))
        cvars = [0, 2]
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + n_scenarios], x[i][cvars]))

        # tests evaluated_g with out
        c.fill(0.0)
        c = self.nlp2.evaluate_c(x, evaluated_g=g, out=c)
        self.assertIsInstance(c, BlockVector)
        self.assertEqual(c.nblocks, 2 * n_scenarios)
        self.assertEqual(c.size, n_scenarios * (nci + nz))
        cvars = [0, 2]
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(c[i], ci))
            self.assertTrue(np.allclose(c[i + n_scenarios], x[i][cvars]))

    def test_evaluate_d(self):

        instance = self.scenarios2['s0']
        n_scenarios = len(self.scenarios2)
        ndi = instance.nd
        xi = instance.x_init()
        di = instance.evaluate_d(xi)

        x = self.nlp2.x_init()
        d = self.nlp2.evaluate_d(x)
        self.assertIsInstance(d, BlockVector)
        self.assertEqual(d.nblocks, n_scenarios)
        self.assertEqual(d.size, ndi * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(di, d[i]))

        # test out
        d.fill(0.0)
        self.nlp2.evaluate_d(x, out=d)
        self.assertIsInstance(d, BlockVector)
        self.assertEqual(d.nblocks, n_scenarios)
        self.assertEqual(d.size, ndi * n_scenarios)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(di, d[i]))

        # test evaluated_g
        g = self.nlp2.evaluate_g(x)
        d = self.nlp2.evaluate_d(x, evaluated_g=g)
        self.assertIsInstance(d, BlockVector)
        self.assertEqual(d.nblocks, n_scenarios)
        self.assertEqual(d.size, n_scenarios * n_scenarios)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(d[i], di))

        # test evaluated_g
        d.fill(0.0)
        self.nlp2.evaluate_d(x, evaluated_g=g, out=d)
        self.assertIsInstance(d, BlockVector)
        self.assertEqual(d.nblocks, n_scenarios)
        self.assertEqual(d.size, n_scenarios * n_scenarios)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(d[i], di))

    def test_jacobian_g(self):

        nz = len(self.complicated_vars_ids)
        nxi = nz + self.G.shape[1]
        ngi = nz + self.A.shape[0]
        Ji = BlockMatrix(2, 2)
        Ji[0, 0] = coo_matrix(self.A)
        B1 = np.zeros((nz, self.A.shape[1]))
        B2 = np.zeros((nz, nz))
        for i, v in enumerate(self.complicated_vars_ids):
            B1[i, v] = -1.0
            B2[i, i] = 1.0
        Ji[1, 0] = coo_matrix(B1)
        Ji[1, 1] = coo_matrix(B2)
        dense_Ji = Ji.todense()

        x = self.nlp.create_vector_x()
        jac_g = self.nlp.jacobian_g(x)

        total_nx = nxi * self.n_scenarios + nz
        total_ng = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_g.shape, (total_ng, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_gi = jac_g[i, i].todense()
            self.assertTrue(np.allclose(jac_gi, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = identity(nz)
        Ai_[0, 0] = empty_matrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -identity(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_g[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_g[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test out
        # change g values
        for i in range(self.n_scenarios):
            jac_g[i, i] *= 2.0
            jac_gi = jac_g[i, i].todense()
            self.assertTrue(np.allclose(jac_gi, 2*dense_Ji))
        self.nlp.jacobian_g(x, out=jac_g)

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_gi = jac_g[i, i].todense()
            self.assertTrue(np.allclose(jac_gi, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = identity(nz)
        Ai_[0, 0] = empty_matrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -identity(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_g[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_g[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test flattened vector
        jac_g = self.nlp.jacobian_g(x.flatten())

        total_nx = nxi * self.n_scenarios + nz
        total_ng = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_g.shape, (total_ng, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_gi = jac_g[i, i].todense()
            self.assertTrue(np.allclose(jac_gi, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = identity(nz)
        Ai_[0, 0] = empty_matrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -identity(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_g[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_g[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test nlp2
        instance = self.scenarios2['s0']
        n_scenarios = len(self.scenarios2)
        xi = instance.x_init()
        Jci = instance.jacobian_g(xi)

        x = self.nlp2.x_init()
        Jc = self.nlp2.jacobian_g(x)
        self.assertIsInstance(Jc, BlockMatrix)
        self.assertEqual(Jc.bshape, (2 * n_scenarios, n_scenarios + 1))
        AB = self.nlp2.coupling_matrix()
        for i in range(n_scenarios):
            AB[i, i] = AB[i, i].tocoo()
        AB[n_scenarios, n_scenarios] = AB[n_scenarios, n_scenarios].tocoo()

        for i in range(n_scenarios):
            self.assertIsInstance(Jc[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[i, i].row, Jci.row))
            self.assertTrue(np.allclose(Jc[i, i].col, Jci.col))
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data))

            # check Ai
            self.assertIsInstance(Jc[n_scenarios + i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].row,
                                        AB[i, i].row))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].col,
                                        AB[i, i].col))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].data,
                                        AB[i, i].data))

            # check Bi
            coo_identity = Jc[n_scenarios + i, n_scenarios].tocoo()
            self.assertTrue(np.allclose(coo_identity.row,
                                        AB[n_scenarios, n_scenarios].row))
            self.assertTrue(np.allclose(coo_identity.col,
                                        AB[n_scenarios, n_scenarios].col))
            self.assertTrue(np.allclose(coo_identity.data,
                                        AB[n_scenarios, n_scenarios].data))

        # test flattened
        Jc = self.nlp2.jacobian_g(x.flatten())
        self.assertIsInstance(Jc, BlockMatrix)
        self.assertEqual(Jc.bshape, (2 * n_scenarios, n_scenarios + 1))
        AB = self.nlp2.coupling_matrix()
        for i in range(n_scenarios):
            AB[i, i] = AB[i, i].tocoo()
        AB[n_scenarios, n_scenarios] = AB[n_scenarios, n_scenarios].tocoo()

        for i in range(n_scenarios):
            self.assertIsInstance(Jc[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[i, i].row, Jci.row))
            self.assertTrue(np.allclose(Jc[i, i].col, Jci.col))
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data))

            # check Ai
            self.assertIsInstance(Jc[n_scenarios + i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].row,
                                        AB[i, i].row))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].col,
                                        AB[i, i].col))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].data,
                                        AB[i, i].data))

            # check Bi

            coo_identity = Jc[n_scenarios + i, n_scenarios].tocoo()
            self.assertTrue(np.allclose(coo_identity.row,
                                        AB[n_scenarios, n_scenarios].row))
            self.assertTrue(np.allclose(coo_identity.col,
                                        AB[n_scenarios, n_scenarios].col))
            self.assertTrue(np.allclose(coo_identity.data,
                                        AB[n_scenarios, n_scenarios].data))

        # test out
        for i in range(n_scenarios):
            Jc[i, i] *= 2.0
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data * 2.0))

        self.nlp2.jacobian_g(x, out=Jc)
        self.assertIsInstance(Jc, BlockMatrix)
        self.assertEqual(Jc.bshape, (2 * n_scenarios, n_scenarios + 1))
        AB = self.nlp2.coupling_matrix()
        for i in range(n_scenarios):
            AB[i, i] = AB[i, i].tocoo()

        for i in range(n_scenarios):
            self.assertIsInstance(Jc[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[i, i].row, Jci.row))
            self.assertTrue(np.allclose(Jc[i, i].col, Jci.col))
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data))

    def test_jacobian_c(self):

        nz = len(self.complicated_vars_ids)
        nxi = nz + self.G.shape[1]
        ngi = nz + self.A.shape[0]
        Ji = BlockMatrix(2, 2)
        Ji[0, 0] = coo_matrix(self.A)
        B1 = np.zeros((nz, self.A.shape[1]))
        B2 = np.zeros((nz, nz))
        for i, v in enumerate(self.complicated_vars_ids):
            B1[i, v] = -1.0
            B2[i, i] = 1.0
        Ji[1, 0] = coo_matrix(B1)
        Ji[1, 1] = coo_matrix(B2)
        dense_Ji = Ji.todense()

        x = self.nlp.create_vector_x()
        jac_c = self.nlp.jacobian_c(x)

        total_nx = nxi * self.n_scenarios + nz
        total_nc = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_c.shape, (total_nc, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_ci = jac_c[i, i].todense()
            self.assertTrue(np.allclose(jac_ci, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = identity(nz)
        Ai_[0, 0] = empty_matrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -identity(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_c[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_c[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test out
        # change g values
        for i in range(self.n_scenarios):
            jac_c[i, i] *= 2.0
            jac_ci = jac_c[i, i].todense()
            self.assertTrue(np.allclose(jac_ci, 2 * dense_Ji))
        self.nlp.jacobian_c(x, out=jac_c)

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_ci = jac_c[i, i].todense()
            self.assertTrue(np.allclose(jac_ci, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = identity(nz)
        Ai_[0, 0] = empty_matrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -identity(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_c[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_c[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test flattened vector
        jac_g = self.nlp.jacobian_c(x.flatten())

        total_nx = nxi * self.n_scenarios + nz
        total_nc = (ngi + nz) * self.n_scenarios
        self.assertEqual(jac_c.shape, (total_nc, total_nx))

        # check block jacobians
        for i in range(self.n_scenarios):
            jac_ci = jac_c[i, i].todense()
            self.assertTrue(np.allclose(jac_ci, dense_Ji))

        # check coupling jacobians
        Ai_ = BlockMatrix(1, 2)
        Ai_[0, 1] = identity(nz)
        Ai_[0, 0] = empty_matrix(nz, self.G.shape[1])
        Ai_ = Ai_.todense()
        Bi_ = -identity(nz).todense()
        for i in range(self.n_scenarios):
            Ai = jac_c[self.n_scenarios + i, i]
            self.assertTrue(np.allclose(Ai.todense(), Ai_))
            Bi = jac_c[self.n_scenarios + i, self.n_scenarios]
            self.assertTrue(np.allclose(Bi.todense(), Bi_))

        # test nlp2
        instance = self.scenarios2['s0']
        n_scenarios = len(self.scenarios2)
        xi = instance.x_init()
        Jci = instance.jacobian_c(xi)

        x = self.nlp2.x_init()
        Jc = self.nlp2.jacobian_c(x)
        self.assertIsInstance(Jc, BlockMatrix)
        self.assertEqual(Jc.bshape, (2 * n_scenarios, n_scenarios + 1))
        AB = self.nlp2.coupling_matrix()
        for i in range(n_scenarios):
            AB[i, i] = AB[i, i].tocoo()
        AB[n_scenarios, n_scenarios] = AB[n_scenarios, n_scenarios].tocoo()
        for i in range(n_scenarios):
            self.assertIsInstance(Jc[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[i, i].row, Jci.row))
            self.assertTrue(np.allclose(Jc[i, i].col, Jci.col))
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data))

            # check Ai
            self.assertIsInstance(Jc[n_scenarios + i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].row,
                                        AB[i, i].row))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].col,
                                        AB[i, i].col))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].data,
                                        AB[i, i].data))

            # check Bi
            #self.assertIsInstance(Jc[n_scenarios + i, n_scenarios], coo_matrix)
            coo_identity = Jc[n_scenarios + i, n_scenarios].tocoo()
            self.assertTrue(np.allclose(coo_identity.row,
                                        AB[n_scenarios, n_scenarios].row))
            self.assertTrue(np.allclose(coo_identity.col,
                                        AB[n_scenarios, n_scenarios].col))
            self.assertTrue(np.allclose(coo_identity.data,
                                        AB[n_scenarios, n_scenarios].data))

        # test flattened
        Jc = self.nlp2.jacobian_c(x.flatten())
        self.assertIsInstance(Jc, BlockMatrix)
        self.assertEqual(Jc.bshape, (2 * n_scenarios, n_scenarios + 1))
        AB = self.nlp2.coupling_matrix()
        for i in range(n_scenarios):
            AB[i, i] = AB[i, i].tocoo()
        AB[n_scenarios, n_scenarios] = AB[n_scenarios, n_scenarios].tocoo()

        for i in range(n_scenarios):
            self.assertIsInstance(Jc[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[i, i].row, Jci.row))
            self.assertTrue(np.allclose(Jc[i, i].col, Jci.col))
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data))

            # check Ai
            self.assertIsInstance(Jc[n_scenarios + i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].row,
                                        AB[i, i].row))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].col,
                                        AB[i, i].col))
            self.assertTrue(np.allclose(Jc[n_scenarios + i, i].data,
                                        AB[i, i].data))

            # check Bi
            #self.assertIsInstance(Jc[n_scenarios + i, n_scenarios], coo_matrix)
            coo_identity = Jc[n_scenarios + i, n_scenarios].tocoo()
            self.assertTrue(np.allclose(coo_identity.row,
                                        AB[n_scenarios, n_scenarios].row))
            self.assertTrue(np.allclose(coo_identity.col,
                                        AB[n_scenarios, n_scenarios].col))
            self.assertTrue(np.allclose(coo_identity.data,
                                        AB[n_scenarios, n_scenarios].data))

        # test out
        for i in range(n_scenarios):
            Jc[i, i] *= 2.0
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data * 2.0))

        self.nlp2.jacobian_c(x, out=Jc)
        self.assertIsInstance(Jc, BlockMatrix)
        self.assertEqual(Jc.bshape, (2 * n_scenarios, n_scenarios + 1))
        AB = self.nlp2.coupling_matrix()
        for i in range(n_scenarios):
            AB[i, i] = AB[i, i].tocoo()

        for i in range(n_scenarios):
            self.assertIsInstance(Jc[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jc[i, i].row, Jci.row))
            self.assertTrue(np.allclose(Jc[i, i].col, Jci.col))
            self.assertTrue(np.allclose(Jc[i, i].data, Jci.data))

    def test_jacobian_d(self):

        instance = self.scenarios2['s0']
        n_scenarios = len(self.scenarios2)
        xi = instance.x_init()
        Jdi = instance.jacobian_d(xi)

        x = self.nlp2.x_init()
        Jd = self.nlp2.jacobian_d(x)
        self.assertIsInstance(Jd, BlockMatrix)
        self.assertEqual(Jd.bshape, (n_scenarios, n_scenarios))
        for i in range(n_scenarios):
            self.assertIsInstance(Jd[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jd[i, i].row, Jdi.row))
            self.assertTrue(np.allclose(Jd[i, i].col, Jdi.col))
            self.assertTrue(np.allclose(Jd[i, i].data, Jdi.data))

        for i in range(n_scenarios):
            Jd[i, i] *= 2.0
            self.assertTrue(np.allclose(Jd[i, i].data, Jdi.data*2.0))

        self.nlp2.jacobian_d(x, out=Jd)
        for i in range(n_scenarios):
            self.assertIsInstance(Jd[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jd[i, i].row, Jdi.row))
            self.assertTrue(np.allclose(Jd[i, i].col, Jdi.col))
            self.assertTrue(np.allclose(Jd[i, i].data, Jdi.data))

        Jd = self.nlp2.jacobian_d(x.flatten())
        self.assertIsInstance(Jd, BlockMatrix)
        self.assertEqual(Jd.bshape, (n_scenarios, n_scenarios))
        for i in range(n_scenarios):
            self.assertIsInstance(Jd[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jd[i, i].row, Jdi.row))
            self.assertTrue(np.allclose(Jd[i, i].col, Jdi.col))
            self.assertTrue(np.allclose(Jd[i, i].data, Jdi.data))

        for i in range(n_scenarios):
            Jd[i, i] *= 2.0
            self.assertTrue(np.allclose(Jd[i, i].data, Jdi.data * 2.0))

        self.nlp2.jacobian_d(x.flatten(), out=Jd)
        for i in range(n_scenarios):
            self.assertIsInstance(Jd[i, i], coo_matrix)
            self.assertTrue(np.allclose(Jd[i, i].row, Jdi.row))
            self.assertTrue(np.allclose(Jd[i, i].col, Jdi.col))
            self.assertTrue(np.allclose(Jd[i, i].data, Jdi.data))

    def test_hessian(self):

        nz = len(self.complicated_vars_ids)
        Hi = BlockSymMatrix(2)
        Hi[0, 0] = coo_matrix(self.G)
        Hi[1, 1] = empty_matrix(nz, nz) # this is because of the way the test problem was setup

        Hi = Hi.todense()
        x = self.nlp.create_vector_x()
        y = self.nlp.create_vector_y()
        H = self.nlp.hessian_lag(x, y)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(H[i, i].todense(), Hi))
        self.assertTrue(np.allclose(H[self.n_scenarios, self.n_scenarios].todense(),
                                    empty_matrix(nz, nz).todense()))

        # test out
        # change g values
        for i in range(self.n_scenarios):
            H[i, i] *= 2.0
            Hj = H[i, i].todense()
            self.assertTrue(np.allclose(Hj, 2.0 * Hi))
        self.nlp.hessian_lag(x, y, out=H)

        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(H[i, i].todense(), Hi))
        self.assertTrue(np.allclose(H[self.n_scenarios, self.n_scenarios].todense(),
                                    empty_matrix(nz, nz).todense()))

        H = self.nlp.hessian_lag(x.flatten(), y)
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(H[i, i].todense(), Hi))
        self.assertTrue(np.allclose(H[self.n_scenarios, self.n_scenarios].todense(),
                                    empty_matrix(nz, nz).todense()))

        H = self.nlp.hessian_lag(x.flatten(), y.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(H[i, i].todense(), Hi))
        self.assertTrue(np.allclose(H[self.n_scenarios, self.n_scenarios].todense(),
                                    empty_matrix(nz, nz).todense()))

        H = self.nlp.hessian_lag(x, y.flatten())
        for i in range(self.n_scenarios):
            self.assertTrue(np.allclose(H[i, i].todense(), Hi))
        self.assertTrue(np.allclose(H[self.n_scenarios, self.n_scenarios].todense(),
                                    empty_matrix(nz, nz).todense()))

    def test_expansion_matrix_xl(self):

        instance = self.scenarios2['s0']
        xli = instance.xl(condensed=True)
        Pxli = instance.expansion_matrix_xl()
        all_xli = Pxli * xli
        xxi = np.copy(instance.xl())
        xxi[xxi == -np.inf] = 0
        self.assertTrue(np.allclose(all_xli, xxi))

        lower_x = self.nlp2.xl(condensed=True)
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        Pxl = self.nlp2.expansion_matrix_xl()
        self.assertIsInstance(Pxl, BlockMatrix)
        self.assertFalse(Pxl.has_empty_rows())
        self.assertFalse(Pxl.has_empty_cols())
        self.assertEqual(Pxl.bshape, (n_scenarios + 1, n_scenarios + 1))


        # for i in range(n_scenarios):
        #     print(i, lower_x[i])
        # for i in range(n_scenarios):
        #     for j in range(n_scenarios):
        #         if Pxl[i, j] is not None:
        #             print(i, j, Pxl[i, j].shape)

        all_xl = Pxl * lower_x
        self.assertIsInstance(all_xl, BlockVector)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(xxi, all_xl[i]))
        self.assertTrue(np.allclose(np.zeros(nz), all_xl[n_scenarios]))

    def expansion_matrix_xu(self):

        instance = self.scenarios2['s0']
        xui = instance.xu(condensed=True)
        Pxui = instance.expansion_matrix_xu()
        all_xui = Pxui * xui
        xxi = np.copy(instance.xu())
        xxi[xxi == np.inf] = 0

        upper_x = self.nlp2.xu(condensed=True)
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        Pxu = self.nlp2.expansion_matrix_xu()
        self.assertIsInstance(Pxu, BlockMatrix)
        self.assertFalse(Pxu.has_empty_rows())
        self.assertFalse(Pxu.has_empty_cols())
        self.assertEqual(Pxu.bshape, (n_scenarios + 1, n_scenarios + 1))

        all_xu = Pxu * upper_x
        self.assertIsInstance(all_xu, BlockVector)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(xxi, all_xu[i]))
        self.assertTrue(np.allclose(np.zeros(nz), all_xu[n_scenarios]))

    def test_expansion_matrix_dl(self):

        instance = self.scenarios2['s0']
        dli = instance.dl(condensed=True)
        Pdli = instance.expansion_matrix_dl()
        all_dli = Pdli * dli
        ddi = np.copy(instance.dl())
        ddi[ddi == -np.inf] = 0
        self.assertTrue(np.allclose(all_dli, ddi))

        lower_d = self.nlp2.dl(condensed=True)
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        Pdl = self.nlp2.expansion_matrix_dl()
        self.assertIsInstance(Pdl, BlockMatrix)
        self.assertFalse(Pdl.has_empty_rows())
        self.assertFalse(Pdl.has_empty_cols())
        self.assertEqual(Pdl.bshape, (n_scenarios, n_scenarios))

        all_dl = Pdl * lower_d
        self.assertIsInstance(all_dl, BlockVector)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(ddi, all_dl[i]))

    def test_expansion_matrix_du(self):

        instance = self.scenarios2['s0']
        dui = instance.du(condensed=True)
        Pdui = instance.expansion_matrix_du()
        all_dui = Pdui * dui
        ddi = np.copy(instance.du())
        ddi[ddi == np.inf] = 0
        self.assertTrue(np.allclose(all_dui, ddi))

        upper_d = self.nlp2.du(condensed=True)
        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        Pdu = self.nlp2.expansion_matrix_du()
        self.assertIsInstance(Pdu, BlockMatrix)
        self.assertFalse(Pdu.has_empty_rows())
        self.assertFalse(Pdu.has_empty_cols())
        self.assertEqual(Pdu.bshape, (n_scenarios, n_scenarios))

        all_du = Pdu * upper_d
        self.assertIsInstance(all_du, BlockVector)
        for i in range(n_scenarios):
            self.assertTrue(np.allclose(ddi, all_du[i]))

    def test_coupling_matrix(self):

        n_scenarios = len(self.scenarios2)
        nz = len(self.complicated_vars_ids2)
        AB = self.nlp2.coupling_matrix()
        self.assertIsInstance(AB, BlockMatrix)
        self.assertFalse(AB.has_empty_rows())
        self.assertFalse(AB.has_empty_cols())
        self.assertEqual(AB.bshape, (n_scenarios+1, n_scenarios+1))
        x = self.nlp2.create_vector_x()
        x.fill(1.0)
        zs = AB * x
        for i in range(n_scenarios):
            self.assertEqual(zs[i].size, nz)
        self.assertEqual(zs[n_scenarios].size, nz)


