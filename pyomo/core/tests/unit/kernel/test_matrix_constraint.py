#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

import pickle

import pyutilib.th as unittest
import pyomo.kernel as pmo
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     ICategorizedObjectContainer)
from pyomo.core.kernel.homogeneous_container import \
    IHomogeneousContainer
from pyomo.core.kernel.tuple_container import TupleContainer
from pyomo.core.kernel.constraint import (IConstraint,
                                          constraint,
                                          constraint_dict,
                                          constraint_tuple,
                                          constraint_list)
from pyomo.core.kernel.matrix_constraint import \
    (matrix_constraint,
     _MatrixConstraintData)
from pyomo.core.kernel.variable import (variable,
                                        variable_list)
from pyomo.core.kernel.parameter import parameter
from pyomo.core.kernel.expression import expression
from pyomo.core.kernel.block import (block,
                                     block_list)

try:
    import numpy
    has_numpy = True
except:
    has_numpy = False

try:
    import scipy
    has_scipy = True
    _scipy_ver = tuple(int(_) for _ in scipy.version.version.split('.')[:2])
except:
    has_scipy = False
    _scipy_ver = (0,0)


def _create_variable_list(size, **kwds):
    assert size > 0
    vlist = variable_list()
    for i in range(size):
        vlist.append(variable(**kwds))
    return vlist


@unittest.skipUnless(has_numpy and has_scipy,
                     "NumPy or SciPy is not available")
class Test_matrix_constraint(unittest.TestCase):

    def test_pprint(self):
        # Not really testing what the output is, just that
        # an error does not occur. The pprint functionality
        # is still in the early stages.
        m,n = 3,2
        vlist = _create_variable_list(2)
        A = numpy.random.rand(m,n)
        ctuple = matrix_constraint(A,
                                   lb=1,
                                   ub=2,
                                   x=vlist)
        pmo.pprint(ctuple)
        b = block()
        b.c = ctuple
        pmo.pprint(ctuple)
        pmo.pprint(b)
        m = block()
        m.b = b
        pmo.pprint(ctuple)
        pmo.pprint(b)
        pmo.pprint(m)

    def test_ctype(self):
        ctuple = matrix_constraint(numpy.random.rand(3,3))
        self.assertIs(ctuple.ctype, IConstraint)
        self.assertIs(type(ctuple), matrix_constraint)
        self.assertIs(type(ctuple)._ctype, IConstraint)
        self.assertIs(ctuple[0].ctype, IConstraint)
        self.assertIs(type(ctuple[0])._ctype, IConstraint)

    def test_pickle(self):
        vlist = _create_variable_list(3)
        ctuple = matrix_constraint(
            numpy.array([[0,0,0],[0,0,0]]),
            x=vlist)
        self.assertTrue((ctuple.lb == -numpy.inf).all())
        self.assertTrue((ctuple.ub == numpy.inf).all())
        self.assertTrue((ctuple.equality == False).all())
        self.assertEqual(ctuple.parent, None)
        ctuple_up = pickle.loads(
            pickle.dumps(ctuple))
        self.assertTrue((ctuple_up.lb == -numpy.inf).all())
        self.assertTrue((ctuple_up.ub == numpy.inf).all())
        self.assertTrue((ctuple_up.equality == False).all())
        self.assertEqual(ctuple_up.parent, None)
        b = block()
        b.ctuple = ctuple
        self.assertIs(ctuple.parent, b)
        bup = pickle.loads(
            pickle.dumps(b))
        ctuple_up = bup.ctuple
        self.assertTrue((ctuple_up.lb == -numpy.inf).all())
        self.assertTrue((ctuple_up.ub == numpy.inf).all())
        self.assertTrue((ctuple_up.equality == False).all())
        self.assertIs(ctuple_up.parent, bup)

    def test_init(self):
        vlist = _create_variable_list(3, value=1.0)
        ctuple = matrix_constraint(numpy.zeros((3,3)),
                                   x=vlist)
        self.assertEqual(len(ctuple), 3)
        self.assertTrue(ctuple.parent is None)
        self.assertEqual(ctuple.ctype, IConstraint)
        self.assertTrue((ctuple.lb == -numpy.inf).all())
        self.assertTrue((ctuple.ub == numpy.inf).all())
        self.assertTrue((ctuple.equality == False).all())
        for c in ctuple:
            self.assertEqual(c.lb, -numpy.inf)
            self.assertEqual(c.ub, numpy.inf)
            self.assertEqual(c.equality, False)
            self.assertEqual(c(), 0)
            self.assertEqual(c.slack, float('inf'))
            self.assertEqual(c.lslack, float('inf'))
            self.assertEqual(c.uslack, float('inf'))
            self.assertEqual(c.has_lb(), False)
            self.assertEqual(c.has_ub(), False)

        vlist = _create_variable_list(3, value=3)
        A = numpy.ones((2,3))
        ctuple = matrix_constraint(A,
                                   lb=0,
                                   ub=2,
                                   x=vlist)
        self.assertEqual(len(ctuple), 2)
        self.assertTrue((ctuple.lb == 0).all())
        self.assertTrue((ctuple.ub == 2).all())
        self.assertTrue((ctuple.equality == False).all())
        for c in ctuple:
            self.assertEqual(len(list(c.terms)), 3)
            self.assertEqual(c.lb, 0)
            self.assertEqual(c.body(), 9)
            self.assertEqual(c(), 9)
            self.assertEqual(c.ub, 2)


        ctuple = matrix_constraint(A,
                                   rhs=1,
                                   x=vlist)
        self.assertEqual(len(ctuple), 2)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(len(list(c.terms)), 3)
            self.assertEqual(c.lb, 1)
            self.assertEqual(c.body(), 9)
            self.assertEqual(c(), 9)
            self.assertEqual(c.ub, 1)
            self.assertEqual(c.rhs, 1)

        # can't use both lb and rhs
        with self.assertRaises(ValueError):
            matrix_constraint(A,
                              lb=0,
                              rhs=0,
                              x=vlist)
        # can't use both ub and rhs
        with self.assertRaises(ValueError):
            matrix_constraint(A,
                              ub=0,
                              rhs=0,
                              x=vlist)

    def test_type(self):
        A = numpy.ones((2,3))
        ctuple = matrix_constraint(A)
        self.assertTrue(isinstance(ctuple, ICategorizedObject))
        self.assertTrue(isinstance(ctuple, ICategorizedObjectContainer))
        self.assertTrue(isinstance(ctuple, IHomogeneousContainer))
        self.assertTrue(isinstance(ctuple, TupleContainer))
        self.assertTrue(isinstance(ctuple, constraint_tuple))
        self.assertTrue(isinstance(ctuple, matrix_constraint))
        self.assertTrue(isinstance(ctuple[0], ICategorizedObject))
        self.assertTrue(isinstance(ctuple[0], IConstraint))
        self.assertTrue(isinstance(ctuple[0], _MatrixConstraintData))

    def test_active(self):
        A = numpy.ones((2,2))
        ctuple = matrix_constraint(A)
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            self.assertEqual(c.active, True)
        ctuple.deactivate()
        self.assertEqual(ctuple.active, False)
        for c in ctuple:
            self.assertEqual(c.active, True)
        ctuple.deactivate(shallow=False)
        self.assertEqual(ctuple.active, False)
        for c in ctuple:
            self.assertEqual(c.active, False)
        ctuple[0].activate()
        self.assertEqual(ctuple.active, False)
        self.assertEqual(ctuple[0].active, True)
        self.assertEqual(ctuple[1].active, False)
        ctuple.activate()
        self.assertEqual(ctuple.active, True)
        self.assertEqual(ctuple[0].active, True)
        self.assertEqual(ctuple[1].active, False)
        ctuple.activate(shallow=False)
        self.assertEqual(ctuple.active, True)
        for c in ctuple:
            self.assertEqual(c.active, True)

        b = block()
        self.assertEqual(b.active, True)
        b.deactivate()
        self.assertEqual(b.active, False)
        b.c = ctuple
        self.assertEqual(ctuple.active, True)
        self.assertEqual(b.active, False)
        ctuple.deactivate()
        self.assertEqual(ctuple.active, False)
        self.assertEqual(b.active, False)
        b.activate()
        self.assertEqual(ctuple.active, False)
        self.assertEqual(b.active, True)
        b.activate(shallow=False)
        self.assertEqual(ctuple.active, True)
        self.assertEqual(b.active, True)
        b.deactivate(shallow=False)
        self.assertEqual(ctuple.active, False)
        self.assertEqual(b.active, False)

    def test_index(self):
        A = numpy.ones((4,5))
        ctuple = matrix_constraint(A)
        for i, c in enumerate(ctuple):
            self.assertEqual(c.index, i)

    @unittest.skipIf(_scipy_ver < (1,1),
                     "csr_matrix.reshape only available in scipy >= 1.1")
    def test_A(self):
        A = numpy.ones((4,5))

        # sparse
        c = matrix_constraint(A)
        self.assertEqual(c.A.shape, A.shape)
        self.assertTrue((c.A == A).all())
        self.assertEqual(c.sparse, True)
        with self.assertRaises(ValueError):
            c.A.data[0] = 2
        with self.assertRaises(ValueError):
            c.A.indices[0] = 2
        with self.assertRaises(ValueError):
            c.A.indptr[0] = 2
        cA = c.A
        cA.shape = (5,4)
        # the shape of c.A should not be changed
        self.assertEqual(c.A.shape, (4,5))

        # dense
        c = matrix_constraint(A, sparse=False)
        self.assertEqual(c.A.shape, A.shape)
        self.assertTrue((c.A == A).all())
        self.assertEqual(c.sparse, False)
        with self.assertRaises(ValueError):
            c.A[0,0] = 2
        cA = c.A
        cA.shape = (5,4)
        # the shape of c.A should not be changed
        self.assertEqual(c.A.shape, (4,5))

    def test_x(self):
        A = numpy.ones((4,5))
        ctuple = matrix_constraint(A)
        self.assertEqual(ctuple.x, None)
        for c in ctuple:
            with self.assertRaises(ValueError):
                list(c.terms)
        vlist = _create_variable_list(5)
        ctuple.x = vlist
        self.assertEqual(len(ctuple.x), 5)
        self.assertEqual(len(ctuple.x), len(vlist))
        self.assertIsNot(ctuple.x, len(vlist))
        for i, v in enumerate(ctuple.x):
            self.assertIs(v, vlist[i])

        ctuple = matrix_constraint(A, x=vlist)
        self.assertEqual(len(ctuple.x), 5)
        self.assertEqual(len(ctuple.x), len(vlist))
        self.assertIsNot(ctuple.x, len(vlist))
        for i, v in enumerate(ctuple.x):
            self.assertIs(v, vlist[i])

        vlist = _create_variable_list(4)
        with self.assertRaises(ValueError):
            ctuple = matrix_constraint(A, x=vlist)
        ctuple = matrix_constraint(A)
        with self.assertRaises(ValueError):
            ctuple.x = vlist

    def test_bad_shape(self):
        A = numpy.array([[1,2,3],[1,2,3]])
        matrix_constraint(A)
        A = scipy.sparse.csr_matrix(numpy.array([[1,2,3],[1,2,3]]))
        matrix_constraint(A)
        A = numpy.array([1,2,3])
        with self.assertRaises(ValueError):
            matrix_constraint(A)
        A = numpy.ones((2,2,2))
        with self.assertRaises(ValueError):
            matrix_constraint(A)
        A = [1,2,3]
        with self.assertRaises(AttributeError):
            matrix_constraint(A)

    def test_equality(self):
        A = numpy.ones((5,4))
        ctuple = matrix_constraint(A, rhs=1)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(c.lb, 1)
            self.assertEqual(c.ub, 1)
            self.assertEqual(c.equality, True)

        ctuple = matrix_constraint(A, rhs=numpy.zeros(5))
        self.assertTrue((ctuple.lb == 0).all())
        self.assertTrue((ctuple.ub == 0).all())
        self.assertTrue((ctuple.rhs == 0).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(c.lb, 0)
            self.assertEqual(c.ub, 0)
            self.assertEqual(c.equality, True)

        # can not set when equality is True
        with self.assertRaises(ValueError):
            ctuple.lb = 2
        with self.assertRaises(ValueError):
            ctuple.lb = ctuple.lb
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.lb = 2
        # can not set when equality is True
        with self.assertRaises(ValueError):
            ctuple.ub = 2
        with self.assertRaises(ValueError):
            ctuple.ub = ctuple.ub
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.ub = 2

        ctuple.equality = False
        # can not get when equality is False
        with self.assertRaises(ValueError):
            ctuple.rhs

        self.assertTrue((ctuple.lb == 0).all())
        self.assertTrue((ctuple.ub == 0).all())
        self.assertTrue((ctuple.equality == False).all())
        for c in ctuple:
            self.assertEqual(c.lb, 0)
            self.assertEqual(c.ub, 0)
            self.assertEqual(c.equality, False)

        # can not set to True, must set rhs to a value
        with self.assertRaises(ValueError):
            ctuple.equality = True
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.equality = True

        ctuple.rhs = 3
        self.assertTrue((ctuple.lb == 3).all())
        self.assertTrue((ctuple.ub == 3).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(c.lb, 3)
            self.assertEqual(c.ub, 3)
            self.assertEqual(c.equality, True)

        with self.assertRaises(ValueError):
            ctuple.rhs = None
        for c in ctuple:
            with self.assertRaises(ValueError):
                ctuple.rhs = None
        self.assertTrue((ctuple.lb == 3).all())
        self.assertTrue((ctuple.ub == 3).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(c.lb, 3)
            self.assertEqual(c.ub, 3)
            self.assertEqual(c.equality, True)

        ctuple.equality = False
        self.assertTrue((ctuple.lb == 3).all())
        self.assertTrue((ctuple.ub == 3).all())
        self.assertTrue((ctuple.equality == False).all())
        for c in ctuple:
            self.assertEqual(c.lb, 3)
            self.assertEqual(c.ub, 3)
            self.assertEqual(c.equality, False)

        ctuple.rhs = 4
        self.assertTrue((ctuple.lb == 4).all())
        self.assertTrue((ctuple.ub == 4).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(c.lb, 4)
            self.assertEqual(c.ub, 4)
            self.assertEqual(c.equality, True)

        for c in ctuple:
            c.equality = False
        self.assertTrue((ctuple.lb == 4).all())
        self.assertTrue((ctuple.ub == 4).all())
        self.assertTrue((ctuple.equality == False).all())
        for c in ctuple:
            self.assertEqual(c.lb, 4)
            self.assertEqual(c.ub, 4)
            self.assertEqual(c.equality, False)


    def test_nondata_bounds(self):
        A = numpy.ones((5,4))
        ctuple = matrix_constraint(A, rhs=1)

        eL = expression()
        eU = expression()
        with self.assertRaises(ValueError):
            ctuple.rhs = eL
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.rhs = eL
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())

        vL = variable()
        vU = variable()
        with self.assertRaises(ValueError):
            ctuple.rhs = vL
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.rhs = vL
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())

        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            ctuple.rhs = vL
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.rhs = vL
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())

        # the fixed status of a variable
        # does not change this restriction
        vL.fixed = True
        vU.fixed = True
        with self.assertRaises(ValueError):
            ctuple.rhs = vL
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.rhs = vL
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())

        p = parameter(value=0)
        with self.assertRaises(ValueError):
            ctuple.rhs = p
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.rhs = p
        self.assertTrue((ctuple.rhs == 1).all())
        self.assertTrue((ctuple.equality == True).all())


        ctuple.equality = False
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())

        eL = expression()
        eU = expression()
        with self.assertRaises(ValueError):
            ctuple.lb = eL
        with self.assertRaises(ValueError):
            ctuple.ub = eU
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.lb = eL
            with self.assertRaises(ValueError):
                c.ub = eU
            with self.assertRaises(ValueError):
                c.bounds = (eL, eU)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())

        vL = variable()
        vU = variable()
        with self.assertRaises(ValueError):
            ctuple.lb = vL
        with self.assertRaises(ValueError):
            ctuple.ub = vU
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.lb = vL
            with self.assertRaises(ValueError):
                c.ub = vU
            with self.assertRaises(ValueError):
                c.bounds = (vL, vU)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())

        vL.value = 1.0
        vU.value = 1.0
        with self.assertRaises(ValueError):
            ctuple.lb = vL
        with self.assertRaises(ValueError):
            ctuple.ub = vU
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.lb = vL
            with self.assertRaises(ValueError):
                c.ub = vU
            with self.assertRaises(ValueError):
                c.bounds = (vL, vU)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())

        # the fixed status of a variable
        # does not change this restriction
        vL.fixed = True
        vU.fixed = True
        with self.assertRaises(ValueError):
            ctuple.lb = vL
        with self.assertRaises(ValueError):
            ctuple.ub = vU
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.lb = vL
            with self.assertRaises(ValueError):
                c.ub = vU
            with self.assertRaises(ValueError):
                c.bounds = (vL, vU)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())

        p = parameter(value=0)
        with self.assertRaises(ValueError):
            ctuple.lb = p
        with self.assertRaises(ValueError):
            ctuple.ub = p
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.lb = p
            with self.assertRaises(ValueError):
                c.ub = p
            with self.assertRaises(ValueError):
                c.bounds = (p, p)
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())

    def test_data_bounds(self):
        A = numpy.ones((5,4))
        ctuple = matrix_constraint(A)
        self.assertTrue((ctuple.lb == -numpy.inf).all())
        self.assertTrue((ctuple.ub == numpy.inf).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for c in ctuple:
            self.assertEqual(c.lb, -numpy.inf)
            self.assertEqual(c.ub, numpy.inf)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        ctuple.lb = 1
        ctuple.ub = 2
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 2).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for c in ctuple:
            self.assertEqual(c.lb, 1)
            self.assertEqual(c.ub, 2)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        with self.assertRaises(ValueError):
            ctuple.lb = range(5)
        with self.assertRaises(ValueError):
            ctuple.lb = numpy.array(range(6))
        with self.assertRaises(ValueError):
            ctuple.ub = range(5)
        with self.assertRaises(ValueError):
            ctuple.ub = numpy.array(range(6))
        with self.assertRaises(ValueError):
            ctuple.equality = True
        for c in ctuple:
            with self.assertRaises(ValueError):
                ctuple.equality = True

        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 2).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for c in ctuple:
            self.assertEqual(c.lb, 1)
            self.assertEqual(c.ub, 2)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        for c in ctuple:
            c.bounds = (-1, 1)
        self.assertTrue((ctuple.lb == -1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for c in ctuple:
            self.assertEqual(c.lb, -1)
            self.assertEqual(c.ub, 1)
            self.assertEqual(c.bounds, (-1, 1))
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        ctuple.lb = lb_ = -numpy.array(range(5))
        ctuple.ub = ub_ = numpy.array(range(5))
        self.assertTrue((ctuple.lb == lb_).all())
        self.assertTrue((ctuple.ub == ub_).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for i, c in enumerate(ctuple):
            self.assertEqual(c.lb, -i)
            self.assertEqual(c.ub, i)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        for c in ctuple:
            c.lb = None
            c.ub = None
        self.assertTrue((ctuple.lb == -numpy.inf).all())
        self.assertTrue((ctuple.ub == numpy.inf).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for c in ctuple:
            self.assertEqual(c.lb, -numpy.inf)
            self.assertEqual(c.ub, numpy.inf)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        for i, c in enumerate(ctuple):
            c.lb = -i
            c.ub = i
        self.assertTrue((ctuple.lb == lb_).all())
        self.assertTrue((ctuple.ub == ub_).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for i, c in enumerate(ctuple):
            self.assertEqual(c.lb, -i)
            self.assertEqual(c.ub, i)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        ctuple.lb = None
        ctuple.ub = None
        self.assertTrue((ctuple.lb == -numpy.inf).all())
        self.assertTrue((ctuple.ub == numpy.inf).all())
        self.assertTrue((ctuple.equality == False).all())
        with self.assertRaises(ValueError):
            ctuple.rhs
        for c in ctuple:
            self.assertEqual(c.lb, -numpy.inf)
            self.assertEqual(c.ub, numpy.inf)
            self.assertEqual(c.equality, False)
            with self.assertRaises(ValueError):
                c.rhs

        ctuple.rhs = 1
        self.assertTrue((ctuple.lb == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.ub == 1).all())
        self.assertTrue((ctuple.equality == True).all())
        for c in ctuple:
            self.assertEqual(c.lb, 1)
            self.assertEqual(c.ub, 1)
            self.assertEqual(c.rhs, 1)
            self.assertEqual(c.equality, True)

        rhs_ = numpy.array(range(5))
        for i, c in enumerate(ctuple):
            c.rhs = i
        self.assertTrue((ctuple.lb == rhs_).all())
        self.assertTrue((ctuple.ub == rhs_).all())
        self.assertTrue((ctuple.ub == rhs_).all())
        self.assertTrue((ctuple.equality == True).all())
        for i, c in enumerate(ctuple):
            self.assertEqual(c.lb, i)
            self.assertEqual(c.ub, i)
            self.assertEqual(c.rhs, i)
            self.assertEqual(c.equality, True)

        with self.assertRaises(ValueError):
            ctuple.rhs = None
        with self.assertRaises(ValueError):
            ctuple.rhs = range(5)
        with self.assertRaises(ValueError):
            ctuple.rhs = numpy.array(range(6))
        for c in ctuple:
            with self.assertRaises(ValueError):
                c.rhs = None
        self.assertTrue((ctuple.lb == rhs_).all())
        self.assertTrue((ctuple.ub == rhs_).all())
        self.assertTrue((ctuple.ub == rhs_).all())
        self.assertTrue((ctuple.equality == True).all())
        for i, c in enumerate(ctuple):
            self.assertEqual(c.lb, i)
            self.assertEqual(c.ub, i)
            self.assertEqual(c.rhs, i)
            self.assertEqual(c.equality, True)

    def test_call(self):
        vlist = _create_variable_list(3)
        vlist[0].value = 1
        vlist[1].value = 0
        vlist[2].value = 3
        A = numpy.ones((3,3))
        ctuple = matrix_constraint(A, x=vlist)
        self.assertTrue((ctuple() == 4).all())
        self.assertEqual(ctuple[0](), 4)
        self.assertEqual(ctuple[1](), 4)
        self.assertEqual(ctuple[2](), 4)
        A[:,0] = 0
        A[:,2] = 2
        ctuple = matrix_constraint(A, x=vlist)
        vlist[2].value = 4
        self.assertTrue((ctuple() == 8).all())
        self.assertEqual(ctuple[0](), 8)
        self.assertEqual(ctuple[1](), 8)
        self.assertEqual(ctuple[2](), 8)

        A = numpy.random.rand(4,3)
        ctuple = matrix_constraint(A, x=vlist)
        vlist[1].value = 2
        cvals = numpy.array([ctuple[0](),
                             ctuple[1](),
                             ctuple[2](),
                             ctuple[3]()])
        self.assertTrue((ctuple() == cvals).all())

        vlist[1].value = None
        with self.assertRaises(ValueError):
            ctuple()
        with self.assertRaises(ValueError):
            ctuple(exception=True)
        self.assertIs(ctuple(exception=False), None)
        for c in ctuple:
            with self.assertRaises(ValueError):
                c()
            with self.assertRaises(ValueError):
                c(exception=True)
            self.assertIs(c(exception=False), None)

        ctuple.x = None
        with self.assertRaises(ValueError):
            ctuple()
        with self.assertRaises(ValueError):
            ctuple(exception=True)
        with self.assertRaises(ValueError):
            ctuple(exception=False)
        for c in ctuple:
            with self.assertRaises(ValueError):
                c()
            with self.assertRaises(ValueError):
                c(exception=True)
            with self.assertRaises(ValueError):
                c(exception=False)

    def test_slack(self):
        vlist = _create_variable_list(3)
        vlist[0].value = 1
        vlist[1].value = 0
        vlist[2].value = 3
        A = numpy.ones((3,3))
        ctuple = matrix_constraint(A, x=vlist)
        self.assertTrue((ctuple() == 4).all())
        self.assertEqual(ctuple[0](), 4)
        self.assertEqual(ctuple[1](), 4)
        self.assertEqual(ctuple[2](), 4)
        A[:,0] = 0
        A[:,2] = 2
        ctuple = matrix_constraint(A, x=vlist)
        vlist[2].value = 4
        self.assertTrue((ctuple() == 8).all())
        self.assertEqual(ctuple[0](), 8)
        self.assertEqual(ctuple[1](), 8)
        self.assertEqual(ctuple[2](), 8)

        A = numpy.random.rand(4,3)
        ctuple = matrix_constraint(A, x=vlist)
        vlist[1].value = 2
        cvals = numpy.array([ctuple[0](),
                             ctuple[1](),
                             ctuple[2](),
                             ctuple[3]()])
        self.assertTrue((ctuple() == cvals).all())

    def test_slack_methods(self):
        x = variable(value=2)
        L = 1
        U = 5
        A = numpy.array([[1]])

        cE = matrix_constraint(A, x=[x],
                               rhs=L)
        x.value = 4
        self.assertEqual(cE[0].body(), 4)
        self.assertEqual(cE[0].slack, -3)
        self.assertEqual(cE[0].lslack, 3)
        self.assertEqual(cE[0].uslack, -3)
        self.assertEqual(cE.slack[0], -3)
        self.assertEqual(cE.lslack[0], 3)
        self.assertEqual(cE.uslack[0], -3)
        x.value = 6
        self.assertEqual(cE[0].body(), 6)
        self.assertEqual(cE[0].slack, -5)
        self.assertEqual(cE[0].lslack, 5)
        self.assertEqual(cE[0].uslack, -5)
        self.assertEqual(cE.slack[0], -5)
        self.assertEqual(cE.lslack[0], 5)
        self.assertEqual(cE.uslack[0], -5)
        x.value = 0
        self.assertEqual(cE[0].body(), 0)
        self.assertEqual(cE[0].slack, -1)
        self.assertEqual(cE[0].lslack, -1)
        self.assertEqual(cE[0].uslack, 1)
        self.assertEqual(cE.slack[0], -1)
        self.assertEqual(cE.lslack[0], -1)
        self.assertEqual(cE.uslack[0], 1)
        x.value = None
        with self.assertRaises(ValueError):
            cE[0].body()
        with self.assertRaises(ValueError):
            cE[0].body(exception=True)
        self.assertEqual(cE[0].body(exception=False), None)
        self.assertEqual(cE[0].slack, None)
        self.assertEqual(cE[0].lslack, None)
        self.assertEqual(cE[0].uslack, None)
        self.assertEqual(cE.slack, None)
        self.assertEqual(cE.lslack, None)
        self.assertEqual(cE.uslack, None)

        cE = matrix_constraint(A, x=[x],
                               rhs=U)
        x.value = 4
        self.assertEqual(cE[0].body(), 4)
        self.assertEqual(cE[0].slack, -1)
        self.assertEqual(cE[0].lslack, -1)
        self.assertEqual(cE[0].uslack, 1)
        self.assertEqual(cE.slack[0], -1)
        self.assertEqual(cE.lslack[0], -1)
        self.assertEqual(cE.uslack[0], 1)
        x.value = 6
        self.assertEqual(cE[0].body(), 6)
        self.assertEqual(cE[0].slack, -1)
        self.assertEqual(cE[0].lslack, 1)
        self.assertEqual(cE[0].uslack, -1)
        self.assertEqual(cE.slack[0], -1)
        self.assertEqual(cE.lslack[0], 1)
        self.assertEqual(cE.uslack[0], -1)
        x.value = 0
        self.assertEqual(cE[0].body(), 0)
        self.assertEqual(cE[0].slack, -5)
        self.assertEqual(cE[0].lslack, -5)
        self.assertEqual(cE[0].uslack, 5)
        self.assertEqual(cE.slack[0], -5)
        self.assertEqual(cE.lslack[0], -5)
        self.assertEqual(cE.uslack[0], 5)

        cL = matrix_constraint(A, x=[x],
                               lb=L)
        x.value = 4
        self.assertEqual(cL[0].body(), 4)
        self.assertEqual(cL[0].slack, 3)
        self.assertEqual(cL[0].lslack, 3)
        self.assertEqual(cL[0].uslack, float('inf'))
        self.assertEqual(cL.slack[0], 3)
        self.assertEqual(cL.lslack[0], 3)
        self.assertEqual(cL.uslack[0], float('inf'))
        x.value = 6
        self.assertEqual(cL[0].body(), 6)
        self.assertEqual(cL[0].slack, 5)
        self.assertEqual(cL[0].lslack, 5)
        self.assertEqual(cL[0].uslack, float('inf'))
        self.assertEqual(cL.slack[0], 5)
        self.assertEqual(cL.lslack[0], 5)
        self.assertEqual(cL.uslack[0], float('inf'))
        x.value = 0
        self.assertEqual(cL[0].body(), 0)
        self.assertEqual(cL[0].slack, -1)
        self.assertEqual(cL[0].lslack, -1)
        self.assertEqual(cL[0].uslack, float('inf'))
        self.assertEqual(cL.slack[0], -1)
        self.assertEqual(cL.lslack[0], -1)
        self.assertEqual(cL.uslack[0], float('inf'))

        cL = matrix_constraint(A, x=[x],
                               lb=float('-inf'))
        x.value = 4
        self.assertEqual(cL[0].body(), 4)
        self.assertEqual(cL[0].slack, float('inf'))
        self.assertEqual(cL[0].lslack, float('inf'))
        self.assertEqual(cL[0].uslack, float('inf'))
        self.assertEqual(cL.slack[0], float('inf'))
        self.assertEqual(cL.lslack[0], float('inf'))
        self.assertEqual(cL.uslack[0], float('inf'))
        x.value = 6
        self.assertEqual(cL[0].body(), 6)
        self.assertEqual(cL[0].slack, float('inf'))
        self.assertEqual(cL[0].lslack, float('inf'))
        self.assertEqual(cL[0].uslack, float('inf'))
        self.assertEqual(cL.slack[0], float('inf'))
        self.assertEqual(cL.lslack[0], float('inf'))
        self.assertEqual(cL.uslack[0], float('inf'))
        x.value = 0
        self.assertEqual(cL[0].body(), 0)
        self.assertEqual(cL[0].slack, float('inf'))
        self.assertEqual(cL[0].lslack, float('inf'))
        self.assertEqual(cL[0].uslack, float('inf'))
        self.assertEqual(cL.slack[0], float('inf'))
        self.assertEqual(cL.lslack[0], float('inf'))
        self.assertEqual(cL.uslack[0], float('inf'))

        cU = matrix_constraint(A, x=[x],
                               ub=U)
        x.value = 4
        self.assertEqual(cU[0].body(), 4)
        self.assertEqual(cU[0].slack, 1)
        self.assertEqual(cU[0].lslack, float('inf'))
        self.assertEqual(cU[0].uslack, 1)
        self.assertEqual(cU.slack[0], 1)
        self.assertEqual(cU.lslack[0], float('inf'))
        self.assertEqual(cU.uslack[0], 1)
        x.value = 6
        self.assertEqual(cU[0].body(), 6)
        self.assertEqual(cU[0].slack, -1)
        self.assertEqual(cU[0].lslack, float('inf'))
        self.assertEqual(cU[0].uslack, -1)
        self.assertEqual(cU.slack[0], -1)
        self.assertEqual(cU.lslack[0], float('inf'))
        self.assertEqual(cU.uslack[0], -1)
        x.value = 0
        self.assertEqual(cU[0].body(), 0)
        self.assertEqual(cU[0].slack, 5)
        self.assertEqual(cU[0].lslack, float('inf'))
        self.assertEqual(cU[0].uslack, 5)
        self.assertEqual(cU.slack[0], 5)
        self.assertEqual(cU.lslack[0], float('inf'))
        self.assertEqual(cU.uslack[0], 5)

        cU = matrix_constraint(A, x=[x],
                               ub=float('inf'))
        x.value = 4
        self.assertEqual(cU[0].body(), 4)
        self.assertEqual(cU[0].slack, float('inf'))
        self.assertEqual(cU[0].lslack, float('inf'))
        self.assertEqual(cU[0].uslack, float('inf'))
        self.assertEqual(cU.slack[0], float('inf'))
        self.assertEqual(cU.lslack[0], float('inf'))
        self.assertEqual(cU.uslack[0], float('inf'))
        x.value = 6
        self.assertEqual(cU[0].body(), 6)
        self.assertEqual(cU[0].slack, float('inf'))
        self.assertEqual(cU[0].lslack, float('inf'))
        self.assertEqual(cU[0].uslack, float('inf'))
        self.assertEqual(cU.slack[0], float('inf'))
        self.assertEqual(cU.lslack[0], float('inf'))
        self.assertEqual(cU.uslack[0], float('inf'))
        x.value = 0
        self.assertEqual(cU[0].body(), 0)
        self.assertEqual(cU[0].slack, float('inf'))
        self.assertEqual(cU[0].lslack, float('inf'))
        self.assertEqual(cU[0].uslack, float('inf'))
        self.assertEqual(cU.slack[0], float('inf'))
        self.assertEqual(cU.lslack[0], float('inf'))
        self.assertEqual(cU.uslack[0], float('inf'))

        cR = matrix_constraint(A, x=[x],
                               lb=L, ub=U)
        x.value = 4
        self.assertEqual(cR[0].body(), 4)
        self.assertEqual(cR[0].slack, 1)
        self.assertEqual(cR[0].lslack, 3)
        self.assertEqual(cR[0].uslack, 1)
        self.assertEqual(cR.slack[0], 1)
        self.assertEqual(cR.lslack[0], 3)
        self.assertEqual(cR.uslack[0], 1)
        x.value = 6
        self.assertEqual(cR[0].body(), 6)
        self.assertEqual(cR[0].slack, -1)
        self.assertEqual(cR[0].lslack, 5)
        self.assertEqual(cR[0].uslack, -1)
        self.assertEqual(cR.slack[0], -1)
        self.assertEqual(cR.lslack[0], 5)
        self.assertEqual(cR.uslack[0], -1)
        x.value = 0
        self.assertEqual(cR[0].body(), 0)
        self.assertEqual(cR[0].slack, -1)
        self.assertEqual(cR[0].lslack, -1)
        self.assertEqual(cR[0].uslack, 5)
        self.assertEqual(cR.slack[0], -1)
        self.assertEqual(cR.lslack[0], -1)
        self.assertEqual(cR.uslack[0], 5)

        cR = matrix_constraint(A, x=[x])
        x.value = 4
        self.assertEqual(cR[0].body(), 4)
        self.assertEqual(cR[0].slack, float('inf'))
        self.assertEqual(cR[0].lslack, float('inf'))
        self.assertEqual(cR[0].uslack, float('inf'))
        self.assertEqual(cR.slack[0], float('inf'))
        self.assertEqual(cR.lslack[0], float('inf'))
        self.assertEqual(cR.uslack[0], float('inf'))
        x.value = 6
        self.assertEqual(cR[0].body(), 6)
        self.assertEqual(cR[0].slack, float('inf'))
        self.assertEqual(cR[0].lslack, float('inf'))
        self.assertEqual(cR[0].uslack, float('inf'))
        self.assertEqual(cR.slack[0], float('inf'))
        self.assertEqual(cR.lslack[0], float('inf'))
        self.assertEqual(cR.uslack[0], float('inf'))
        x.value = 0
        self.assertEqual(cR[0].body(), 0)
        self.assertEqual(cR[0].slack, float('inf'))
        self.assertEqual(cR[0].lslack, float('inf'))
        self.assertEqual(cR[0].uslack, float('inf'))
        self.assertEqual(cR.slack[0], float('inf'))
        self.assertEqual(cR.lslack[0], float('inf'))
        self.assertEqual(cR.uslack[0], float('inf'))

        cR = matrix_constraint(A, x=[x],
                               lb=float('-inf'),
                               ub=float('inf'))
        x.value = 4
        self.assertEqual(cR[0].body(), 4)
        self.assertEqual(cR[0].slack, float('inf'))
        self.assertEqual(cR[0].lslack, float('inf'))
        self.assertEqual(cR[0].uslack, float('inf'))
        self.assertEqual(cR.slack[0], float('inf'))
        self.assertEqual(cR.lslack[0], float('inf'))
        self.assertEqual(cR.uslack[0], float('inf'))
        x.value = 6
        self.assertEqual(cR[0].body(), 6)
        self.assertEqual(cR[0].slack, float('inf'))
        self.assertEqual(cR[0].lslack, float('inf'))
        self.assertEqual(cR[0].uslack, float('inf'))
        self.assertEqual(cR.slack[0], float('inf'))
        self.assertEqual(cR.lslack[0], float('inf'))
        self.assertEqual(cR.uslack[0], float('inf'))
        x.value = 0
        self.assertEqual(cR[0].body(), 0)
        self.assertEqual(cR[0].slack, float('inf'))
        self.assertEqual(cR[0].lslack, float('inf'))
        self.assertEqual(cR[0].uslack, float('inf'))
        self.assertEqual(cR.slack[0], float('inf'))
        self.assertEqual(cR.lslack[0], float('inf'))
        self.assertEqual(cR.uslack[0], float('inf'))

    def test_canonical_form_sparse(self):
        A = numpy.array([[0, 2]])
        vlist = _create_variable_list(2)
        ctuple = matrix_constraint(A, x=vlist)
        self.assertEqual(ctuple.sparse, True)
        for c in ctuple:
            self.assertEqual(c._linear_canonical_form, True)
        terms = list(ctuple[0].terms)
        vs,cs = zip(*terms)
        self.assertEqual(len(terms), 1)
        self.assertIs(vs[0], vlist[1])
        self.assertEqual(cs[0], 2)
        repn = ctuple[0].canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], vlist[1])
        self.assertEqual(repn.linear_coefs, (2,))
        self.assertEqual(repn.constant, 0)
        vlist[0].fix(1)
        repn = ctuple[0].canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], vlist[1])
        self.assertEqual(repn.linear_coefs, (2,))
        self.assertEqual(repn.constant, 0)
        vlist[1].fix(2)
        repn = ctuple[0].canonical_form()
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant, 4)
        repn = ctuple[0].canonical_form(compute_values=False)
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant(), 4)

    def test_canonical_form_dense(self):
        A = numpy.array([[0, 2]])
        vlist = _create_variable_list(2)
        ctuple = matrix_constraint(A, x=vlist,
                                   sparse=False)
        self.assertEqual(ctuple.sparse, False)
        for c in ctuple:
            self.assertEqual(c._linear_canonical_form, True)
        terms = list(ctuple[0].terms)
        vs,cs = zip(*terms)
        self.assertEqual(len(terms), 2)
        self.assertIs(vs[0], vlist[0])
        self.assertIs(vs[1], vlist[1])
        self.assertEqual(cs[0], 0)
        self.assertEqual(cs[1], 2)
        repn = ctuple[0].canonical_form()
        self.assertEqual(len(repn.linear_vars), 2)
        self.assertIs(repn.linear_vars[0], vlist[0])
        self.assertIs(repn.linear_vars[1], vlist[1])
        self.assertEqual(repn.linear_coefs, (0, 2))
        self.assertEqual(repn.constant, 0)
        vlist[0].fix(1)
        repn = ctuple[0].canonical_form()
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], vlist[1])
        self.assertEqual(repn.linear_coefs, (2,))
        self.assertEqual(repn.constant, 0)
        vlist[1].fix(2)
        repn = ctuple[0].canonical_form()
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant, 4)
        repn = ctuple[0].canonical_form(compute_values=False)
        self.assertEqual(repn.linear_vars, ())
        self.assertEqual(repn.linear_coefs, ())
        self.assertEqual(repn.constant(), 4)

    def test_preorder_traversal(self):
        A = numpy.ones((3,3))

        m = block()
        m.c = matrix_constraint(A)
        m.v = variable()
        m.V = variable_list()
        m.V.append(variable())
        m.B = block_list()
        m.B.append(block())
        m.B[0].c = matrix_constraint(A)
        m.B[0].v = variable()
        m.B[0].V = variable_list()
        m.B[0].V.append(variable())
        m.b = block()
        m.b.c = constraint_dict()
        m.b.c[None] = matrix_constraint(A)
        m.b.c[1] = matrix_constraint(A)
        m.b.c[2] = constraint()
        m.b.c[3] = constraint_list()

        # don't visit things below a matrix constraint
        # (e.g., cases where we want to handle it in bulk)
        def no_mc_descend(x):
            if isinstance(x, matrix_constraint):
                return False
            return True
        cnt = 0
        for obj in pmo.preorder_traversal(m,
                                          ctype=IConstraint,
                                          descend=no_mc_descend):
            self.assertTrue(type(obj.parent) is not matrix_constraint)
            self.assertTrue((obj.ctype is block._ctype) or \
                            (obj.ctype is constraint._ctype))
            cnt += 1
        self.assertEqual(cnt, 11)

        cnt = 0
        mc_child_cnt = 0
        for obj in pmo.preorder_traversal(m, ctype=IConstraint):
            self.assertTrue((obj.ctype is block._ctype) or \
                            (obj.ctype is constraint._ctype))
            if type(obj.parent) is matrix_constraint:
                mc_child_cnt += 1
            cnt += 1
        self.assertEqual(cnt, 23)
        self.assertEqual(mc_child_cnt, 12)

        self.assertEqual(
            len(list(m.components(ctype=IConstraint))),
            13)

if __name__ == "__main__":
    unittest.main()
