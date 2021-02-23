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
import os

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.asl import AmplInterface
if not AmplInterface.available():
    raise unittest.SkipTest(
        "Pynumero needs the ASL extension to run NLP tests")

import pyomo.environ as pyo
from pyomo.contrib.pynumero.interfaces.ampl_nlp import AslNLP, AmplNLP
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
import tempfile

from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix, \
    build_compression_mask_for_finite_values, full_to_compressed, compressed_to_full


def create_pyomo_model1():
    m = pyo.ConcreteModel()
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    m.S = pyo.Set(initialize=[i+1 for i in range(9)])

    xb = dict()
    xb[1] = (-1,1)
    xb[2] = (-np.inf,2)
    xb[3] = (-3,np.inf)
    xb[4] = (-np.inf, np.inf)
    xb[5] = (-5,5)
    xb[6] = (-np.inf,6)
    xb[7] = (-7,np.inf)
    xb[8] = (-np.inf,np.inf)
    xb[9] = (-9,9)
    m.x = pyo.Var(m.S, initialize=1.0, bounds=lambda m,i: xb[i])

    cb = dict()
    cb[1] = (-1,1)
    cb[2] = (2,2)
    cb[3] = (-3,np.inf)
    cb[4] = (-np.inf, 4)
    cb[5] = (-5,5)
    cb[6] = (-6,-6)
    cb[7] = (-7,np.inf)
    cb[8] = (-np.inf,8)
    cb[9] = (-9,9)

    def c_rule(m,i):
        return (cb[i][0], sum(i*j*m.x[j] for j in m.S), cb[i][1])
    m.c = pyo.Constraint(m.S, rule=c_rule)
    for i in m.S:
        m.dual.set_value(m.c[i], i)

    m.obj = pyo.Objective(expr=sum(i*j*m.x[i]*m.x[j] for i in m.S for j in m.S))

    # add scaling parameters for testing
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.scaling_factor[m.obj] = 5
    for i in m.S:
        m.scaling_factor[m.x[i]] = 2*float(i)
    for i in m.S:
        m.scaling_factor[m.c[i]] = 3*float(i)

    return m

def create_pyomo_model2():
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], domain=pyo.Reals)
    for i in range(1, 4):
        m.x[i].value = i
    m.e1 = pyo.Constraint(expr=m.x[1] ** 2 - m.x[2] - 1 == 0)
    m.e2 = pyo.Constraint(expr=m.x[1] - m.x[3] - 0.5 == 0)
    m.i1 = pyo.Constraint(expr=m.x[1] + m.x[2] <= 100.0)
    m.i2 = pyo.Constraint(expr=m.x[2] + m.x[3] >= -100.0)
    m.i3 = pyo.Constraint(expr=m.x[2] + m.x[3] + m.x[1] >= -500.0)
    m.x[2].setlb(0.0)
    m.x[3].setlb(0.0)
    m.x[2].setub(100.0)
    m.obj = pyo.Objective(expr=m.x[2]**2)
    return m

def execute_extended_nlp_interface(self, anlp):
    self.assertEqual(anlp.n_primals(),9)
    self.assertEqual(anlp.n_constraints(), 9)
    self.assertEqual(anlp.n_eq_constraints(),2)
    self.assertEqual(anlp.n_ineq_constraints(),7)
    self.assertEqual(anlp.nnz_jacobian(), 9*9)
    self.assertEqual(anlp.nnz_jacobian_eq(), 2*9)
    self.assertEqual(anlp.nnz_jacobian_ineq(), 7*9)
    self.assertEqual(anlp.nnz_hessian_lag(), 9*9)

    expected_primals_lb = np.asarray([-1, -np.inf, -3, -np.inf, -5, -np.inf, -7, -np.inf, -9], dtype=np.float64)
    expected_primals_ub = np.asarray([1, 2, np.inf, np.inf, 5, 6, np.inf, np.inf, 9], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_primals_lb, anlp.primals_lb()))
    self.assertTrue(np.array_equal(expected_primals_ub, anlp.primals_ub()))

    expected_constraints_lb = np.asarray([-1, 0, -3, -np.inf, -5, 0, -7, -np.inf, -9], dtype=np.float64)
    expected_constraints_ub = np.asarray([1, 0, np.inf, 4, 5, 0, np.inf, 8, 9], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_constraints_lb, anlp.constraints_lb()))
    self.assertTrue(np.array_equal(expected_constraints_ub, anlp.constraints_ub()))

    expected_ineq_lb = np.asarray([-1, -3, -np.inf, -5, -7, -np.inf, -9], dtype=np.float64)
    expected_ineq_ub = np.asarray([1, np.inf, 4, 5, np.inf, 8, 9], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_ineq_lb, anlp.ineq_lb()))
    self.assertTrue(np.array_equal(expected_ineq_ub, anlp.ineq_ub()))

    expected_init_primals = np.ones(9)
    self.assertTrue(np.array_equal(expected_init_primals, anlp.init_primals()))
    expected_init_duals = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_init_duals, anlp.init_duals()))
    expected_init_duals_ineq = np.asarray([1, 3, 4, 5, 7, 8, 9], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_init_duals_ineq, anlp.init_duals_ineq()))
    expected_init_duals_eq = np.asarray([2, 6], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_init_duals_eq, anlp.init_duals_eq()))

    t = anlp.create_new_vector('primals')
    self.assertTrue(t.size == 9)
    t = anlp.create_new_vector('constraints')
    self.assertTrue(t.size == 9)
    t = anlp.create_new_vector('eq_constraints')
    self.assertTrue(t.size == 2)
    t = anlp.create_new_vector('ineq_constraints')
    self.assertTrue(t.size == 7)
    t = anlp.create_new_vector('duals')
    self.assertTrue(t.size == 9)
    t = anlp.create_new_vector('duals_eq')
    self.assertTrue(t.size == 2)
    t = anlp.create_new_vector('duals_ineq')
    self.assertTrue(t.size == 7)

    expected_primals = [i+1 for i in range(9)]
    new_primals = np.asarray(expected_primals, dtype=np.float64)
    expected_primals = np.asarray(expected_primals, dtype=np.float64)
    anlp.set_primals(new_primals)
    ret = anlp.get_primals()
    self.assertTrue(np.array_equal(new_primals, ret))
    self.assertTrue(np.array_equal(expected_primals, anlp._primals))
    anlp.set_primals(np.ones(9))

    expected_duals = [i+1 for i in range(9)]
    new_duals = np.asarray(expected_duals, dtype=np.float64)
    expected_duals = np.asarray(expected_duals, dtype=np.float64)
    anlp.set_duals(new_duals)
    self.assertTrue(np.array_equal(expected_duals, anlp._duals_full))
    ret = anlp.get_duals()
    self.assertTrue(np.array_equal(new_duals, ret))
    expected_duals_eq = np.asarray([2, 6], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals_eq, anlp._duals_eq))
    expected_duals_ineq = np.asarray([1, 3, 4, 5, 7, 8, 9], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals_ineq, anlp._duals_ineq))
    anlp.set_duals(np.ones(9))

    expected_duals_eq = [i+1 for i in range(2)]
    new_duals_eq = np.asarray(expected_duals_eq, dtype=np.float64)
    anlp.set_duals_eq(new_duals_eq)
    ret = anlp.get_duals_eq()
    self.assertTrue(np.array_equal(new_duals_eq, ret))
    self.assertTrue(np.array_equal(expected_duals_eq, anlp._duals_eq))
    expected_duals = np.asarray([1, 1, 1, 1, 1, 2, 1, 1, 1], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals, anlp._duals_full))
    expected_duals_ineq = np.asarray([1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals_ineq, anlp._duals_ineq))
    anlp.set_duals_eq(np.ones(2))
    expected_duals = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals, anlp._duals_full))

    expected_duals_ineq = [i+1 for i in range(7)]
    new_duals_ineq = np.asarray(expected_duals_ineq, dtype=np.float64)
    anlp.set_duals_ineq(new_duals_ineq)
    ret = anlp.get_duals_ineq()
    self.assertTrue(np.array_equal(new_duals_ineq, ret))
    self.assertTrue(np.array_equal(expected_duals_ineq, anlp._duals_ineq))
    expected_duals = np.asarray([1, 1, 2, 3, 4, 1, 5, 6, 7], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals, anlp._duals_full))
    expected_duals_eq = np.asarray([1, 1], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals_eq, anlp._duals_eq))
    anlp.set_duals_ineq(np.ones(7))
    expected_duals = np.asarray([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_duals, anlp._duals_full))

    # objective function
    expected_objective = sum((i+1)*(j+1) for i in range(9) for j in range(9))
    self.assertEqual(expected_objective, anlp.evaluate_objective())
    # change the value of the primals
    anlp.set_primals(2.0*np.ones(9))
    expected_objective = sum(2.0**2*(i+1)*(j+1) for i in range(9) for j in range(9))
    self.assertEqual(expected_objective, anlp.evaluate_objective())
    anlp.set_primals(np.ones(9))

    # gradient of the objective
    expected_gradient = np.asarray([2*sum((i+1)*(j+1) for j in range(9)) for i in range(9)], dtype=np.float64)
    grad_obj = anlp.evaluate_grad_objective()
    self.assertTrue(np.array_equal(expected_gradient, grad_obj))
    # test inplace
    grad_obj = np.ones(9)
    ret = anlp.evaluate_grad_objective(out=grad_obj)
    self.assertTrue(ret is grad_obj)
    self.assertTrue(np.array_equal(expected_gradient, grad_obj))
    # change the value of the primals
    anlp.set_primals(2.0*np.ones(9))
    expected_gradient = np.asarray([2*2*sum((i+1)*(j+1) for j in range(9)) for i in range(9)], dtype=np.float64)
    grad_obj = np.ones(9)
    anlp.evaluate_grad_objective(out=grad_obj)
    self.assertTrue(np.array_equal(expected_gradient, grad_obj))
    anlp.set_primals(np.ones(9))

    # full constraints
    con = anlp.evaluate_constraints()
    expected_con = np.asarray([45, 88, 3*45, 4*45, 5*45, 276, 7*45, 8*45, 9*45], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_con, con))
    # test inplace
    con = np.zeros(9)
    ret = anlp.evaluate_constraints(out=con)
    self.assertTrue(ret is con)
    self.assertTrue(np.array_equal(expected_con, con))
    # change the value of the primals
    anlp.set_primals(2.0*np.ones(9))
    con = np.zeros(9)
    anlp.evaluate_constraints(out=con)
    expected_con = np.asarray([2*45, 2*(88+2)-2, 2*3*45, 2*4*45, 2*5*45, 2*(276-6)+6, 2*7*45, 2*8*45, 2*9*45], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_con, con))
    anlp.set_primals(np.ones(9))

    # equality constraints
    con_eq = anlp.evaluate_eq_constraints()
    expected_con_eq = np.asarray([88, 276], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_con_eq, con_eq))
    # test inplace
    con_eq = np.zeros(2)
    ret = anlp.evaluate_eq_constraints(out=con_eq)
    self.assertTrue(ret is con_eq)
    self.assertTrue(np.array_equal(expected_con_eq, con_eq))
    # change the value of the primals
    anlp.set_primals(2.0*np.ones(9))
    con_eq = np.zeros(2)
    anlp.evaluate_eq_constraints(out=con_eq)
    expected_con_eq = np.asarray([2*(88+2)-2, 2*(276-6)+6], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_con_eq, con_eq))
    anlp.set_primals(np.ones(9))

    # inequality constraints
    con_ineq = anlp.evaluate_ineq_constraints()
    expected_con_ineq = np.asarray([45, 3*45, 4*45, 5*45, 7*45, 8*45, 9*45], dtype=np.float64)
    self.assertTrue(np.array_equal(expected_con_ineq, con_ineq))
    # test inplace
    con_ineq = np.zeros(7)
    ret = anlp.evaluate_ineq_constraints(out=con_ineq)
    self.assertTrue(ret is con_ineq)
    self.assertTrue(np.array_equal(expected_con_ineq, con_ineq))
    # change the value of the primals
    anlp.set_primals(2.0*np.ones(9))
    con_ineq = np.zeros(7)
    anlp.evaluate_ineq_constraints(out=con_ineq)
    expected_con_ineq = 2.0*expected_con_ineq
    self.assertTrue(np.array_equal(expected_con_ineq, con_ineq))
    anlp.set_primals(np.ones(9))

    # jacobian of all constraints
    jac = anlp.evaluate_jacobian()
    dense_jac = jac.todense()
    expected_jac = [ [(i)*(j) for j in range(1,10)] for i in range(1,10) ]
    expected_jac = np.asarray(expected_jac, dtype=np.float64)
    self.assertTrue(np.array_equal(dense_jac, expected_jac))
    # test inplace
    jac.data = 0*jac.data
    ret = anlp.evaluate_jacobian(out=jac)
    self.assertTrue(ret is jac)
    dense_jac = jac.todense()
    self.assertTrue(np.array_equal(dense_jac, expected_jac))
    # change the value of the primals
    # ToDo: not a great test since this problem is linear
    anlp.set_primals(2.0*np.ones(9))
    anlp.evaluate_jacobian(out=jac)
    dense_jac = jac.todense()
    self.assertTrue(np.array_equal(dense_jac, expected_jac))

    # jacobian of equality constraints
    jac_eq = anlp.evaluate_jacobian_eq()
    dense_jac_eq = jac_eq.todense()
    expected_jac_eq = np.asarray([[2, 4, 6, 8, 10, 12, 14, 16, 18],
                                  [6, 12, 18, 24, 30, 36, 42, 48, 54]], dtype=np.float64)
    self.assertTrue(np.array_equal(dense_jac_eq, expected_jac_eq))
    # test inplace
    jac_eq.data = 0*jac_eq.data
    ret = anlp.evaluate_jacobian_eq(out=jac_eq)
    self.assertTrue(ret is jac_eq)
    dense_jac_eq = jac_eq.todense()
    self.assertTrue(np.array_equal(dense_jac_eq, expected_jac_eq))
    # change the value of the primals
    # ToDo: not a great test since this problem is linear
    anlp.set_primals(2.0*np.ones(9))
    anlp.evaluate_jacobian_eq(out=jac_eq)
    dense_jac_eq = jac_eq.todense()
    self.assertTrue(np.array_equal(dense_jac_eq, expected_jac_eq))

    # jacobian of inequality constraints
    jac_ineq = anlp.evaluate_jacobian_ineq()
    dense_jac_ineq = jac_ineq.todense()
    expected_jac_ineq = [ [(i)*(j) for j in range(1,10)] for i in [1, 3, 4, 5, 7, 8, 9] ]
    expected_jac_ineq = np.asarray(expected_jac_ineq, dtype=np.float64)
    self.assertTrue(np.array_equal(dense_jac_ineq, expected_jac_ineq))
    # test inplace
    jac_ineq.data = 0*jac_ineq.data
    ret = anlp.evaluate_jacobian_ineq(out=jac_ineq)
    self.assertTrue(ret is jac_ineq)
    dense_jac_ineq = jac_ineq.todense()
    self.assertTrue(np.array_equal(dense_jac_ineq, expected_jac_ineq))
    # change the value of the primals
    # ToDo: not a great test since this problem is linear
    anlp.set_primals(2.0*np.ones(9))
    anlp.evaluate_jacobian_ineq(out=jac_ineq)
    dense_jac_ineq = jac_ineq.todense()
    self.assertTrue(np.array_equal(dense_jac_ineq, expected_jac_ineq))

    # hessian
    hess = anlp.evaluate_hessian_lag()
    dense_hess = hess.todense()
    expected_hess = [ [2.0*i*j for j in range(1, 10)] for i in range(1,10) ]
    expected_hess = np.asarray(expected_hess, dtype=np.float64)
    self.assertTrue(np.array_equal(dense_hess, expected_hess))
    # test inplace
    hess.data = np.zeros(len(hess.data))
    ret = anlp.evaluate_hessian_lag(out=hess)
    self.assertTrue(ret is hess)
    dense_hess = hess.todense()
    self.assertTrue(np.array_equal(dense_hess, expected_hess))
    # change the value of the primals
    anlp.set_primals(2.0*np.ones(9))
    anlp.evaluate_hessian_lag(out=hess)
    dense_hess = hess.todense()
    self.assertTrue(np.array_equal(dense_hess, expected_hess))
    # change the value of the obj factor
    anlp.set_obj_factor(2.0)
    hess = anlp.evaluate_hessian_lag()
    dense_hess = hess.todense()
    expected_hess = [ [4.0*i*j for j in range(1, 10)] for i in range(1,10) ]
    expected_hess = np.asarray(expected_hess, dtype=np.float64)
    self.assertTrue(np.array_equal(dense_hess, expected_hess))


class TestAslNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = create_pyomo_model1()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, "Pyomo_TestAslNLP")
        cls.pm.write(cls.filename+'.nl', io_options={"symbolic_solver_labels": True})

    @classmethod
    def tearDownClass(cls):
        # TODO: remove the nl files
        pass

    def test_nlp_interface(self):
        anlp = AslNLP(self.filename)
        execute_extended_nlp_interface(self, anlp)
        # AslNLP does not check suffixes for scaling parameters
        self.assertIsNone(anlp.get_obj_scaling())
        self.assertIsNone(anlp.get_primals_scaling())
        self.assertIsNone(anlp.get_constraints_scaling())
        self.assertIsNone(anlp.get_eq_constraints_scaling())
        self.assertIsNone(anlp.get_ineq_constraints_scaling())
        
class TestAmplNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # test problem
        cls.pm2 = create_pyomo_model2()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, "Pyomo_TestAmplNLP")
        cls.pm2.write(cls.filename+'.nl', io_options={"symbolic_solver_labels": True})
        cls.nlp = AmplNLP(cls.filename+'.nl',
                          row_filename=cls.filename+'.row',
                          col_filename=cls.filename+'.col')

    @classmethod
    def tearDownClass(cls):
        # TODO: remove the nl files
        pass

    def test_names(self):
        # Note: order may not be the same as "expected"
        expected_variable_names = ['x[1]', 'x[2]', 'x[3]']
        variable_names = self.nlp.variable_names()
        self.assertEqual(len(expected_variable_names),len(variable_names))
        for i in range(len(expected_variable_names)):
            self.assertTrue(expected_variable_names[i] in variable_names)

        # Note: order may not be the same as "expected"
        expected_constraint_names = ['e1', 'e2', 'i1', 'i2', 'i3']
        constraint_names = self.nlp.constraint_names()
        self.assertEqual(len(expected_constraint_names),len(constraint_names))
        for i in range(len(expected_constraint_names)):
            self.assertTrue(expected_constraint_names[i] in constraint_names)

        # Note: order may not be the same as "expected"
        expected_eq_constraint_names = ['e1', 'e2']
        eq_constraint_names = self.nlp.eq_constraint_names()
        self.assertEqual(len(expected_eq_constraint_names),len(eq_constraint_names))
        for i in range(len(expected_eq_constraint_names)):
            self.assertTrue(expected_eq_constraint_names[i] in eq_constraint_names)

        # Note: order may not be the same as "expected"
        expected_ineq_constraint_names = ['i1', 'i2', 'i3']
        ineq_constraint_names = self.nlp.ineq_constraint_names()
        self.assertEqual(len(expected_ineq_constraint_names),len(ineq_constraint_names))
        for i in range(len(expected_ineq_constraint_names)):
            self.assertTrue(expected_ineq_constraint_names[i] in ineq_constraint_names)

    def test_idxs(self):
        # Note: order may not be the same as expected
        variable_idxs = list()
        variable_idxs.append(self.nlp.variable_idx('x[1]'))
        variable_idxs.append(self.nlp.variable_idx('x[2]'))
        variable_idxs.append(self.nlp.variable_idx('x[3]'))
        self.assertEqual(sum(variable_idxs), 3)

        # Note: order may not be the same as expected
        constraint_idxs = list()
        constraint_idxs.append(self.nlp.constraint_idx('e1'))
        constraint_idxs.append(self.nlp.constraint_idx('e2'))
        constraint_idxs.append(self.nlp.constraint_idx('i1'))
        constraint_idxs.append(self.nlp.constraint_idx('i2'))
        constraint_idxs.append(self.nlp.constraint_idx('i3'))
        self.assertEqual(sum(constraint_idxs), 10)

        # Note: order may not be the same as expected
        eq_constraint_idxs = list()
        eq_constraint_idxs.append(self.nlp.eq_constraint_idx('e1'))
        eq_constraint_idxs.append(self.nlp.eq_constraint_idx('e2'))
        self.assertEqual(sum(eq_constraint_idxs), 1)

        # Note: order may not be the same as expected
        ineq_constraint_idxs = list()
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('i1'))
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('i2'))
        ineq_constraint_idxs.append(self.nlp.ineq_constraint_idx('i3'))
        self.assertEqual(sum(ineq_constraint_idxs), 3)


class TestPyomoNLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # test problem
        cls.pm = create_pyomo_model1()
        
    @classmethod
    def tearDownClass(cls):
        pass

    def test_nlp_interface(self):
        nlp = PyomoNLP(self.pm)
        execute_extended_nlp_interface(self, nlp)
        self.assertTrue(nlp.pyomo_model() is self.pm)

        self.assertEqual(float(nlp.get_obj_scaling()), 5.0)

        xs = nlp.get_primals_scaling()
        expected_xs = np.asarray([2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0])
        self.assertTrue(np.array_equal(xs, expected_xs))

        cs = nlp.get_constraints_scaling()
        expected_cs = np.asarray([ 3.0,  6.0,  9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0 ])
        self.assertTrue(np.array_equal(cs, expected_cs))

        eqcs = nlp.get_eq_constraints_scaling()
        expected_eqcs = np.asarray([ 6.0, 18.0 ])
        self.assertTrue(np.array_equal(eqcs, expected_eqcs))

        ineqcs = nlp.get_ineq_constraints_scaling()
        expected_ineqcs = np.asarray([ 3.0,  9.0, 12.0, 15.0, 21.0, 24.0, 27.0 ])
        self.assertTrue(np.array_equal(ineqcs, expected_ineqcs))


    def test_indices_methods(self):
        nlp = PyomoNLP(self.pm)

        # get_pyomo_variables
        variables = nlp.get_pyomo_variables()
        expected_ids = [id(self.pm.x[i]) for i in range(1,10)]
        ids = [id(variables[i]) for i in range(9)]
        self.assertTrue(expected_ids == ids)

        variable_names = nlp.variable_names()
        expected_names = [self.pm.x[i].getname() for i in range(1,10)]
        self.assertTrue(variable_names == expected_names)

        # get_pyomo_constraints
        constraints = nlp.get_pyomo_constraints()
        expected_ids = [id(self.pm.c[i]) for i in range(1,10)]
        ids = [id(constraints[i]) for i in range(9)]
        self.assertTrue(expected_ids == ids)

        constraint_names = nlp.constraint_names()
        expected_names = [c.getname() for c in nlp.get_pyomo_constraints()]
        self.assertTrue(constraint_names == expected_names)

        # get_pyomo_equality_constraints
        eq_constraints = nlp.get_pyomo_equality_constraints()
        # 2 and 6 are the equality constraints
        eq_indices = [2, 6] # "indices" here is a bit overloaded
        expected_eq_ids = [id(self.pm.c[i]) for i in eq_indices]
        eq_ids = [id(con) for con in eq_constraints]
        self.assertEqual(eq_ids, expected_eq_ids)

        eq_constraint_names = nlp.equality_constraint_names()
        expected_eq_names = [c.getname(fully_qualified=True)
                for c in nlp.get_pyomo_equality_constraints()]
        self.assertEqual(eq_constraint_names, expected_eq_names)

        # get_pyomo_inequality_constraints
        ineq_constraints = nlp.get_pyomo_inequality_constraints()
        # 1, 3, 4, 5, 7, 8, and 9 are the inequality constraints
        ineq_indices = [1, 3, 4, 5, 7, 8, 9]
        expected_ineq_ids = [id(self.pm.c[i]) for i in ineq_indices]
        ineq_ids = [id(con) for con in ineq_constraints]
        self.assertEqual(eq_ids, expected_eq_ids)

        # get_primal_indices
        expected_primal_indices = [i for i in range(9)]
        self.assertTrue(expected_primal_indices == nlp.get_primal_indices([self.pm.x]))
        expected_primal_indices = [0, 3, 8, 4]
        variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
        self.assertTrue(expected_primal_indices == nlp.get_primal_indices(variables))

        # get_constraint_indices
        expected_constraint_indices = [i for i in range(9)]
        self.assertTrue(expected_constraint_indices == nlp.get_constraint_indices([self.pm.c]))
        expected_constraint_indices = [0, 3, 8, 4]
        constraints = [self.pm.c[1], self.pm.c[4], self.pm.c[9], self.pm.c[5]]
        self.assertTrue(expected_constraint_indices == nlp.get_constraint_indices(constraints))

        # get_equality_constraint_indices
        pyomo_eq_indices = [2, 6]
        with self.assertRaises(KeyError):
            # At least one data object in container is not an equality
            nlp.get_equality_constraint_indices([self.pm.c])
        eq_constraints = [self.pm.c[i] for i in pyomo_eq_indices]
        expected_eq_indices = [0, 1]
        # ^indices in the list of equality constraints
        eq_constraint_indices = nlp.get_equality_constraint_indices(
                eq_constraints)
        self.assertEqual(expected_eq_indices, eq_constraint_indices)

        # get_inequality_constraint_indices
        pyomo_ineq_indices = [1, 3, 4, 5, 7, 9]
        with self.assertRaises(KeyError):
            # At least one data object in container is not an equality
            nlp.get_inequality_constraint_indices([self.pm.c])
        ineq_constraints = [self.pm.c[i] for i in pyomo_ineq_indices]
        expected_ineq_indices = [0, 1, 2, 3, 4, 6]
        # ^indices in the list of equality constraints; didn't include 8
        ineq_constraint_indices = nlp.get_inequality_constraint_indices(
                ineq_constraints)
        self.assertEqual(expected_ineq_indices, ineq_constraint_indices)

        # extract_subvector_grad_objective
        expected_gradient = np.asarray([2*sum((i+1)*(j+1) for j in range(9)) for i in range(9)], dtype=np.float64)
        grad_obj = nlp.extract_subvector_grad_objective([self.pm.x])
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))

        expected_gradient = np.asarray([2*sum((i+1)*(j+1) for j in range(9)) for i in [0, 3, 8, 4]], dtype=np.float64)
        variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
        grad_obj = nlp.extract_subvector_grad_objective(variables)
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))

        # extract_subvector_constraints
        expected_con = np.asarray([45, 88, 3*45, 4*45, 5*45, 276, 7*45, 8*45, 9*45], dtype=np.float64)
        con = nlp.extract_subvector_constraints([self.pm.c])
        self.assertTrue(np.array_equal(expected_con, con))

        expected_con = np.asarray([45, 4*45, 9*45, 5*45], dtype=np.float64)
        constraints = [self.pm.c[1], self.pm.c[4], self.pm.c[9], self.pm.c[5]]
        con = nlp.extract_subvector_constraints(constraints)
        self.assertTrue(np.array_equal(expected_con, con))

        # extract_submatrix_jacobian
        expected_jac = [ [(i)*(j) for j in range(1,10)] for i in range(1,10) ]
        expected_jac = np.asarray(expected_jac, dtype=np.float64)
        jac = nlp.extract_submatrix_jacobian(pyomo_variables=[self.pm.x], pyomo_constraints=[self.pm.c])
        dense_jac = jac.todense()
        self.assertTrue(np.array_equal(dense_jac, expected_jac))

        expected_jac = [ [(i)*(j) for j in [1, 4, 9, 5]] for i in [2, 6, 4] ]
        expected_jac = np.asarray(expected_jac, dtype=np.float64)
        variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
        constraints = [self.pm.c[2], self.pm.c[6], self.pm.c[4]]
        jac = nlp.extract_submatrix_jacobian(pyomo_variables=variables, pyomo_constraints=constraints)
        dense_jac = jac.todense()
        self.assertTrue(np.array_equal(dense_jac, expected_jac))

        # extract_submatrix_hessian_lag
        expected_hess = [ [2.0*i*j for j in range(1, 10)] for i in range(1,10) ]
        expected_hess = np.asarray(expected_hess, dtype=np.float64)
        hess = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=[self.pm.x], pyomo_variables_cols=[self.pm.x])
        dense_hess = hess.todense()
        self.assertTrue(np.array_equal(dense_hess, expected_hess))

        expected_hess = [ [2.0*i*j for j in [1, 4, 9, 5]] for i in [1, 4, 9, 5]]
        expected_hess = np.asarray(expected_hess, dtype=np.float64)
        variables = [self.pm.x[1], self.pm.x[4], self.pm.x[9], self.pm.x[5]]
        hess = nlp.extract_submatrix_hessian_lag(pyomo_variables_rows=variables, pyomo_variables_cols=variables)
        dense_hess = hess.todense()
        self.assertTrue(np.array_equal(dense_hess, expected_hess))

    def test_no_objective(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.c = pyo.Constraint(expr=2.0*m.x>=5)
        with self.assertRaises(NotImplementedError):
            nlp = PyomoNLP(m)

class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pm = create_pyomo_model1()
        temporary_dir = tempfile.mkdtemp()
        cls.filename = os.path.join(temporary_dir, "Pyomo_TestAslNLP")
        cls.pm.write(cls.filename+'.nl', io_options={"symbolic_solver_labels": True})

    @classmethod
    def tearDownClass(cls):
        # TODO: remove the nl files
        pass

    def test_util_maps(self):
        anlp = AslNLP(self.filename)
        full_to_compressed_mask = build_compression_mask_for_finite_values(anlp.primals_lb())

        # test build_bounds_mask - should be the same as above
        self.assertTrue(np.array_equal(full_to_compressed_mask, build_bounds_mask(anlp.primals_lb())))

        expected_compressed_primals_lb = np.asarray([-1, -3, -5, -7, -9], dtype=np.float64)

        # test build_compression_matrix
        C = build_compression_matrix(full_to_compressed_mask)
        compressed_primals_lb = C*anlp.primals_lb()
        self.assertTrue(np.array_equal(expected_compressed_primals_lb, compressed_primals_lb))

        # test full_to_compressed
        compressed_primals_lb = full_to_compressed(anlp.primals_lb(), full_to_compressed_mask)
        self.assertTrue(np.array_equal(expected_compressed_primals_lb, compressed_primals_lb))
        # test in place
        compressed_primals_lb = np.zeros(len(expected_compressed_primals_lb))
        ret = full_to_compressed(anlp.primals_lb(), full_to_compressed_mask, out=compressed_primals_lb)
        self.assertTrue(ret is compressed_primals_lb)
        self.assertTrue(np.array_equal(expected_compressed_primals_lb, compressed_primals_lb))
        
        # test compressed_to_full
        expected_full_primals_lb = np.asarray([-1, -np.inf, -3, -np.inf, -5, -np.inf, -7, -np.inf, -9], dtype=np.float64)
        full_primals_lb = compressed_to_full(compressed_primals_lb, full_to_compressed_mask, default=-np.inf)
        self.assertTrue(np.array_equal(expected_full_primals_lb, full_primals_lb))
        # test in place
        full_primals_lb.fill(0.0)
        ret = compressed_to_full(compressed_primals_lb, full_to_compressed_mask, out=full_primals_lb, default=-np.inf)
        self.assertTrue(ret is full_primals_lb)
        self.assertTrue(np.array_equal(expected_full_primals_lb, full_primals_lb))

        # test no default
        expected_full_primals_lb = np.asarray([-1, np.nan, -3, np.nan, -5, np.nan, -7, np.nan, -9], dtype=np.float64)
        full_primals_lb = compressed_to_full(compressed_primals_lb, full_to_compressed_mask)
        print(expected_full_primals_lb)
        print(full_primals_lb)
        np.testing.assert_array_equal(expected_full_primals_lb, full_primals_lb)
        # test in place no default
        expected_full_primals_lb = np.asarray([-1, 0.0, -3, 0.0, -5, 0.0, -7, 0.0, -9], dtype=np.float64)
        full_primals_lb.fill(0.0)
        ret = compressed_to_full(compressed_primals_lb, full_to_compressed_mask, out=full_primals_lb)
        self.assertTrue(ret is full_primals_lb)
        self.assertTrue(np.array_equal(expected_full_primals_lb, full_primals_lb))


if __name__ == '__main__':
    TestAslNLP.setUpClass()
    t = TestAslNLP()
    t.test_create()
    
