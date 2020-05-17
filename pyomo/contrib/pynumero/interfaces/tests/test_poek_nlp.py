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

from pyomo.contrib.pynumero import numpy_available, scipy_available
if not (numpy_available and scipy_available):
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

import scipy.sparse as sp
import numpy as np

import pyomo.environ as pyo
try:
    import poek as pk
    poek_available=True
except:
    poek_available=False

#from pyomo.opt.base import WriterFactory
from pyomo.contrib.pynumero.interfaces.poeknl_nlp import PoekNL_NLP
#import tempfile

#from scipy.sparse import coo_matrix

#from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix, \
    #build_compression_mask_for_finite_values, full_to_compressed, compressed_to_full


def create_poek_model1(zzz=10):
    m = pk.model()
    #m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    x = dict()
    x[1] = m.variable(lb=-1,      ub=1,      value=1.0)
    x[2] = m.variable(lb=-np.inf, ub=2,      value=1.0)
    x[3] = m.variable(lb=-3,      ub=np.inf, value=1.0)
    x[4] = m.variable(lb=-np.inf, ub=np.inf, value=1.0)
    x[5] = m.variable(lb=-5,      ub=5,      value=1.0)
    x[6] = m.variable(lb=-np.inf, ub=6,      value=1.0)
    x[7] = m.variable(lb=-7,      ub=np.inf, value=1.0)
    x[8] = m.variable(lb=-np.inf, ub=np.inf, value=1.0)
    x[9] = m.variable(lb=-9,      ub=9,      value=1.0)

    for i in range(1,10):
        e = sum(i*j*x[j] for j in range(1,zzz))
        if i==1:
            m.add( pk.inequality(-1, e, 1) )
        elif i==2:
            m.add( e == 2)
        elif i==3:
            m.add( e >= -3)
        elif i==4:
            m.add( e <= 4)
        elif i==5:
            m.add( pk.inequality(-5, e, 5) )
        elif i==6:
            m.add( e == -6)
        elif i==7:
            m.add( e >= -7)
        elif i==8:
            m.add( e <= 8)
        elif i==9:
            m.add( pk.inequality(-9, e, 9) )

    m.add( sum(i*j*x[i]*x[j] for i in range(1,10) for j in range(1,10)))
    #m.display()
    return x, pk.nlp_model(m, "cppad")

def create_poek_model2():
    m = pk.model()

    x = m.variable(index=[1,2,3])
    for i in range(1, 4):
        x[i].value = i

    e1 = m.add( x[1] ** 2 - x[2] - 1 == 0 )
    e2 = m.add( x[1] - x[3] - 0.5 == 0 )
    i1 = m.add( x[1] + x[2] <= 100.0 )
    i2 = m.add( x[2] + x[3] >= -100.0 )
    i3 = m.add( x[2] + x[3] + x[1] >= -500.0 )

    x[2].lb = 0.0
    x[3].lb = 0.0
    x[2].ub = 100.0

    obj = m.add( x[2]**2 )

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

    if False:
        # TODO - Fix dual logic in POEKNL_NLP
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


class TestPoekNL_NLP(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # test problem
        cls.x, cls.pm = create_poek_model1()
        cls.c = {i+1:cls.pm.get_constraint(i) for i in range(cls.pm.num_constraints())}
        
    @classmethod
    def tearDownClass(cls):
        pass

    def test_nlp_interface(self):
        nlp = PoekNL_NLP(self.pm)
        execute_extended_nlp_interface(self, nlp)
        self.assertTrue(nlp.poek_nlpmodel() is self.pm)

    def test_indices_methods(self):
        nlp = PoekNL_NLP(self.pm)

        # get_pyomo_variables
        variables = nlp.get_poek_variables()
        expected_ids = [self.x[i].id for i in range(1,10)]
        ids = [variables[i].id for i in range(9)]
        self.assertTrue(expected_ids == ids)

        variable_names = nlp.variable_names()
        expected_names = [self.x[i].name for i in range(1,10)]
        self.assertTrue(variable_names == expected_names)

        # get_pyomo_constraints
        constraints = nlp.get_poek_constraints()
        expected_ids = [self.pm.get_constraint(i).id for i in range(9)]
        ids = [c.id for c in constraints]
        self.assertTrue(expected_ids == ids)

        constraint_names = nlp.constraint_names()
        constraint_names2 = nlp.constraint_names()
        expected_names = [c.name for c in nlp.get_poek_constraints()]
        self.assertTrue(constraint_names == expected_names)

        # get_primal_indices
        expected_primal_indices = [i for i in range(9)]
        self.assertTrue(expected_primal_indices == nlp.get_primal_indices([self.x[i] for i in self.x]))
        expected_primal_indices = [0, 3, 8, 4]
        variables = [self.x[1], self.x[4], self.x[9], self.x[5]]
        self.assertTrue(expected_primal_indices == nlp.get_primal_indices(variables))

        # get_constraint_indices
        expected_constraint_indices = [i for i in range(9)]
        self.assertTrue(expected_constraint_indices == nlp.get_constraint_indices(nlp.get_poek_constraints()))
        expected_constraint_indices = [0, 3, 8, 4]
        constraints = [nlp.poek_nlpmodel().get_constraint(i) for i in [0,3,8,4]]
        self.assertTrue(expected_constraint_indices == nlp.get_constraint_indices(constraints))

        # extract_subvector_grad_objective
        expected_gradient = np.asarray([2*sum((i+1)*(j+1) for j in range(9)) for i in range(9)], dtype=np.float64)
        grad_obj = nlp.extract_subvector_grad_objective([self.x[i] for i in self.x])
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))

        expected_gradient = np.asarray([2*sum((i+1)*(j+1) for j in range(9)) for i in [0, 3, 8, 4]], dtype=np.float64)
        variables = [self.x[1], self.x[4], self.x[9], self.x[5]]
        grad_obj = nlp.extract_subvector_grad_objective(variables)
        self.assertTrue(np.array_equal(expected_gradient, grad_obj))

        # extract_subvector_constraints
        expected_con = np.asarray([45, 88, 3*45, 4*45, 5*45, 276, 7*45, 8*45, 9*45], dtype=np.float64)
        con = nlp.extract_subvector_constraints([nlp.poek_nlpmodel().get_constraint(i) for i in range(nlp.poek_nlpmodel().num_constraints())])
        self.assertTrue(np.array_equal(expected_con, con))

        expected_con = np.asarray([45, 4*45, 9*45, 5*45], dtype=np.float64)
        constraints = [self.c[1], self.c[4], self.c[9], self.c[5]]
        con = nlp.extract_subvector_constraints(constraints)
        self.assertTrue(np.array_equal(expected_con, con))

        # extract_submatrix_jacobian
        expected_jac = [ [(i)*(j) for j in range(1,10)] for i in range(1,10) ]
        expected_jac = np.asarray(expected_jac, dtype=np.float64)
        jac = nlp.extract_submatrix_jacobian(poek_variables=[self.x[i] for i in range(1,1+len(self.x))], poek_constraints=[self.c[i] for i in range(1,1+len(self.c))])
        dense_jac = jac.todense()
        self.assertTrue(np.array_equal(dense_jac, expected_jac))

        expected_jac = [ [(i)*(j) for j in [1, 4, 9, 5]] for i in [2, 6, 4] ]
        expected_jac = np.asarray(expected_jac, dtype=np.float64)
        variables = [self.x[1], self.x[4], self.x[9], self.x[5]]
        constraints = [self.c[2], self.c[6], self.c[4]]
        jac = nlp.extract_submatrix_jacobian(poek_variables=variables, poek_constraints=constraints)
        dense_jac = jac.todense()
        self.assertTrue(np.array_equal(dense_jac, expected_jac))

        # extract_submatrix_hessian_lag
        expected_hess = [ [2.0*i*j for j in range(1, 10)] for i in range(1,10) ]
        expected_hess = np.asarray(expected_hess, dtype=np.float64)
        hess = nlp.extract_submatrix_hessian_lag(poek_variables_rows=[self.x[i] for i in range(1,1+len(self.x))], poek_variables_cols=[self.x[i] for i in range(1,1+len(self.x))])
        dense_hess = hess.todense()
        self.assertTrue(np.array_equal(dense_hess, expected_hess))

        expected_hess = [ [2.0*i*j for j in [1, 4, 9, 5]] for i in [1, 4, 9, 5]]
        expected_hess = np.asarray(expected_hess, dtype=np.float64)
        variables = [self.x[1], self.x[4], self.x[9], self.x[5]]
        hess = nlp.extract_submatrix_hessian_lag(poek_variables_rows=variables, poek_variables_cols=variables)
        dense_hess = hess.todense()
        self.assertTrue(np.array_equal(dense_hess, expected_hess))

    def test_no_objective(self):
        m = pk.model()
        x = pk.variable()
        m.add( 2.0*x >= 5 )
        with self.assertRaises(NotImplementedError):
            nlp = PoekNL_NLP(m)

if __name__ == "__main__":      #pragma:nocover
    unittest.main()
