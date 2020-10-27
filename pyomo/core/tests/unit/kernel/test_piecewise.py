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
from pyomo.core.tests.unit.kernel.test_dict_container import \
    _TestActiveDictContainerBase
from pyomo.core.tests.unit.kernel.test_list_container import \
    _TestActiveListContainerBase
from pyomo.core.kernel.base import \
    (ICategorizedObject,
     ICategorizedObjectContainer)
from pyomo.core.kernel.heterogeneous_container import \
    IHeterogeneousContainer
from pyomo.core.kernel.block import (IBlock,
                                     block,
                                     block_dict,
                                     block_list)
from pyomo.core.kernel.variable import (variable,
                                        variable_list)
from pyomo.core.kernel.piecewise_library.transforms import \
    (PiecewiseLinearFunction,
     TransformedPiecewiseLinearFunction)
import pyomo.core.kernel.piecewise_library.transforms as \
    transforms
from pyomo.core.kernel.piecewise_library.transforms_nd import \
    (PiecewiseLinearFunctionND,
     TransformedPiecewiseLinearFunctionND)
import pyomo.core.kernel.piecewise_library.transforms_nd as \
    transforms_nd
import pyomo.core.kernel.piecewise_library.util as util

# for the multi-dimensional piecewise tests
_test_v = None
_test_tri = None
_test_values = None
def setUpModule():
    global _test_v
    global _test_tri
    global _test_values
    if util.numpy_available and util.scipy_available:
        _test_v = variable_list(
            variable(lb=i, ub=i+1) for i in range(3))
        _test_tri = util.generate_delaunay(_test_v, num=4)
        _test_values = []
        for _xi in _test_tri.points:
            _test_values.append(sum(_xi))
        _test_values = util.numpy.array(_test_values)

class Test_util(unittest.TestCase):

    def test_is_constant(self):
        self.assertEqual(util.is_constant([]), True)
        self.assertEqual(util.is_constant([1]), True)
        self.assertEqual(util.is_constant([1,2]), False)
        self.assertEqual(util.is_constant([1,1]), True)
        self.assertEqual(util.is_constant([1,2,3]), False)
        self.assertEqual(util.is_constant([2.1,2.1,2.1]), True)
        self.assertEqual(util.is_constant([1,1,3,4]), False)
        self.assertEqual(util.is_constant([1,1,3,3]), False)
        self.assertEqual(util.is_constant([1,1,1,4]), False)
        self.assertEqual(util.is_constant([1,1,1,1]), True)
        self.assertEqual(util.is_constant([-1,1,1,1]), False)
        self.assertEqual(util.is_constant([1,-1,1,1]), False)
        self.assertEqual(util.is_constant([1,1,-1,1]), False)
        self.assertEqual(util.is_constant([1,1,1,-1]), False)

    def test_is_nondecreasing(self):
        self.assertEqual(util.is_nondecreasing([]), True)
        self.assertEqual(util.is_nondecreasing([1]), True)
        self.assertEqual(util.is_nondecreasing([1,2]), True)
        self.assertEqual(util.is_nondecreasing([1,2,3]), True)
        self.assertEqual(util.is_nondecreasing([1,1,3,4]), True)
        self.assertEqual(util.is_nondecreasing([1,1,3,3]), True)
        self.assertEqual(util.is_nondecreasing([1,1,1,4]), True)
        self.assertEqual(util.is_nondecreasing([1,1,1,1]), True)
        self.assertEqual(util.is_nondecreasing([-1,1,1,1]), True)
        self.assertEqual(util.is_nondecreasing([1,-1,1,1]), False)
        self.assertEqual(util.is_nondecreasing([1,1,-1,1]), False)
        self.assertEqual(util.is_nondecreasing([1,1,1,-1]), False)

    def test_is_nonincreasing(self):
        self.assertEqual(util.is_nonincreasing([]), True)
        self.assertEqual(util.is_nonincreasing([1]), True)
        self.assertEqual(util.is_nonincreasing([2,1]), True)
        self.assertEqual(util.is_nonincreasing([3,2,1]), True)
        self.assertEqual(util.is_nonincreasing([4,3,2,1]), True)
        self.assertEqual(util.is_nonincreasing([3,3,1,1]), True)
        self.assertEqual(util.is_nonincreasing([4,1,1,1]), True)
        self.assertEqual(util.is_nonincreasing([1,1,1,1]), True)
        self.assertEqual(util.is_nonincreasing([-1,1,1,1]), False)
        self.assertEqual(util.is_nonincreasing([1,-1,1,1]), False)
        self.assertEqual(util.is_nonincreasing([1,1,-1,1]), False)
        self.assertEqual(util.is_nonincreasing([1,1,1,-1]), True)

    def test_is_positive_power_of_two(self):
        self.assertEqual(util.is_positive_power_of_two(-8), False)
        self.assertEqual(util.is_positive_power_of_two(-4), False)
        self.assertEqual(util.is_positive_power_of_two(-3), False)
        self.assertEqual(util.is_positive_power_of_two(-2), False)
        self.assertEqual(util.is_positive_power_of_two(-1), False)
        self.assertEqual(util.is_positive_power_of_two(0), False)
        self.assertEqual(util.is_positive_power_of_two(1), True)
        self.assertEqual(util.is_positive_power_of_two(2), True)
        self.assertEqual(util.is_positive_power_of_two(3), False)
        self.assertEqual(util.is_positive_power_of_two(4), True)
        self.assertEqual(util.is_positive_power_of_two(5), False)
        self.assertEqual(util.is_positive_power_of_two(6), False)
        self.assertEqual(util.is_positive_power_of_two(7), False)
        self.assertEqual(util.is_positive_power_of_two(8), True)
        self.assertEqual(util.is_positive_power_of_two(15), False)
        self.assertEqual(util.is_positive_power_of_two(16), True)
        self.assertEqual(util.is_positive_power_of_two(31), False)
        self.assertEqual(util.is_positive_power_of_two(32), True)

    def test_log2floor(self):
        self.assertEqual(util.log2floor(1), 0)
        self.assertEqual(util.log2floor(2), 1)
        self.assertEqual(util.log2floor(3), 1)
        self.assertEqual(util.log2floor(4), 2)
        self.assertEqual(util.log2floor(5), 2)
        self.assertEqual(util.log2floor(6), 2)
        self.assertEqual(util.log2floor(7), 2)
        self.assertEqual(util.log2floor(8), 3)
        self.assertEqual(util.log2floor(9), 3)
        self.assertEqual(util.log2floor(2**10), 10)
        self.assertEqual(util.log2floor(2**10 + 1), 10)
        self.assertEqual(util.log2floor(2**20), 20)
        self.assertEqual(util.log2floor(2**20 + 1), 20)
        self.assertEqual(util.log2floor(2**30), 30)
        self.assertEqual(util.log2floor(2**30 + 1), 30)
        self.assertEqual(util.log2floor(2**40), 40)
        self.assertEqual(util.log2floor(2**40 + 1), 40)

    def test_generate_gray_code(self):
        self.assertEqual(util.generate_gray_code(0),
                         [[]])
        self.assertEqual(util.generate_gray_code(1),
                         [[0],[1]])
        self.assertEqual(util.generate_gray_code(2),
                         [[0,0],[0,1],[1,1],[1,0]])
        self.assertEqual(util.generate_gray_code(3),
                         [[0,0,0],
                          [0,0,1],
                          [0,1,1],
                          [0,1,0],
                          [1,1,0],
                          [1,1,1],
                          [1,0,1],
                          [1,0,0]])
        self.assertEqual(util.generate_gray_code(4),
                         [[0, 0, 0, 0],
                          [0, 0, 0, 1],
                          [0, 0, 1, 1],
                          [0, 0, 1, 0],
                          [0, 1, 1, 0],
                          [0, 1, 1, 1],
                          [0, 1, 0, 1],
                          [0, 1, 0, 0],
                          [1, 1, 0, 0],
                          [1, 1, 0, 1],
                          [1, 1, 1, 1],
                          [1, 1, 1, 0],
                          [1, 0, 1, 0],
                          [1, 0, 1, 1],
                          [1, 0, 0, 1],
                          [1, 0, 0, 0]])

    def test_characterize_function(self):
        with self.assertRaises(ValueError):
            util.characterize_function([1,2,-1],
                                       [1,1,1])

        fc, slopes = util.characterize_function([1,2,3],
                                                [1,1,1])
        self.assertEqual(fc, 1) # affine
        self.assertEqual(slopes, [0,0])

        fc, slopes = util.characterize_function([1,2,3],
                                                [1,0,1])
        self.assertEqual(fc, 2) # convex
        self.assertEqual(slopes, [-1,1])

        fc, slopes = util.characterize_function([1,2,3],
                                                [1,2,1])
        self.assertEqual(fc, 3) # concave
        self.assertEqual(slopes, [1,-1])

        fc, slopes = util.characterize_function([1,1,2],
                                                [1,2,1])
        self.assertEqual(fc, 4) # step
        self.assertEqual(slopes, [None,-1])

        fc, slopes = util.characterize_function([1,2,3,4],
                                                [1,2,1,2])
        self.assertEqual(fc, 5) # none of the above
        self.assertEqual(slopes, [1,-1,1])

    @unittest.skipUnless(util.numpy_available and util.scipy_available,
                         "Numpy or Scipy is not available")
    def test_generate_delaunay(self):
        vlist = variable_list()
        vlist.append(variable(lb=0, ub=1))
        vlist.append(variable(lb=1, ub=2))
        vlist.append(variable(lb=2, ub=3))
        if not (util.numpy_available and util.scipy_available):
            with self.assertRaises(ImportError):
                util.generate_delaunay(vlist)
        else:
            tri = util.generate_delaunay(vlist, num=2)
            self.assertTrue(
                isinstance(tri, util.scipy.spatial.Delaunay))
            self.assertEqual(len(tri.simplices), 6)
            self.assertEqual(len(tri.points), 8)

            tri = util.generate_delaunay(vlist, num=3)
            self.assertTrue(
                isinstance(tri, util.scipy.spatial.Delaunay))
            self.assertEqual(len(tri.simplices), 62)
            self.assertEqual(len(tri.points), 27)

        #
        # Check cases where not all variables are bounded
        #
        vlist = variable_list()
        vlist.append(variable(lb=0))
        with self.assertRaises(ValueError):
            util.generate_delaunay(vlist)

        vlist = variable_list()
        vlist.append(variable(ub=0))
        with self.assertRaises(ValueError):
            util.generate_delaunay(vlist)

class Test_piecewise(unittest.TestCase):

    def test_pickle(self):
        for key in transforms.registered_transforms:
            v = variable(lb=1,ub=3)
            p = transforms.piecewise([1,2,3],
                                     [1,2,1],
                                     input=v,
                                     validate=False,
                                     repn=key)
            self.assertEqual(p.parent, None)
            self.assertEqual(p.input.expr.parent, None)
            self.assertIs(p.input.expr, v)
            pup = pickle.loads(
                pickle.dumps(p))
            self.assertEqual(pup.parent, None)
            self.assertEqual(pup.input.expr.parent, None)
            self.assertIsNot(pup.input.expr, v)
            b = block()
            b.v = v
            b.p = p
            self.assertIs(p.parent, b)
            self.assertEqual(p.input.expr.parent, b)
            bup = pickle.loads(
                pickle.dumps(b))
            pup = bup.p
            self.assertIs(pup.parent, bup)
            self.assertEqual(pup.input.expr.parent, bup)
            self.assertIs(pup.input.expr, bup.v)
            self.assertIsNot(pup.input.expr, b.v)

    def test_call(self):

        g = PiecewiseLinearFunction([1],
                                    [0])
        f = TransformedPiecewiseLinearFunction(
            g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 0)
        self.assertIs(type(f(1)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(1.1)

        g = PiecewiseLinearFunction([1,2],
                                    [0,4])
        f = TransformedPiecewiseLinearFunction(
            g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 0)
        self.assertIs(type(f(1)), float)
        self.assertEqual(f(1.5), 2)
        self.assertIs(type(f(1.5)), float)
        self.assertEqual(f(2), 4)
        self.assertIs(type(f(2)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(2.1)

        # step function
        g = PiecewiseLinearFunction([1,1],
                                    [0,1])
        f = TransformedPiecewiseLinearFunction(
            g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 0)
        self.assertIs(type(f(1)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(1.1)

        g = PiecewiseLinearFunction([1,2,3],
                                    [1,2,1])
        f = TransformedPiecewiseLinearFunction(
            g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertIs(type(f(1)), float)
        self.assertEqual(f(1.5), 1.5)
        self.assertIs(type(f(1.5)), float)
        self.assertEqual(f(2), 2)
        self.assertIs(type(f(2)), float)
        self.assertEqual(f(2.5), 1.5)
        self.assertIs(type(f(2.5)), float)
        self.assertEqual(f(3), 1)
        self.assertIs(type(f(3)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

        # step function
        g = PiecewiseLinearFunction([1,2,2,3],
                                    [1,2,3,4])
        f = TransformedPiecewiseLinearFunction(
            g, require_bounded_input_variable=False)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertIs(type(f(1)), float)
        self.assertEqual(f(1.5), 1.5)
        self.assertIs(type(f(1.5)), float)
        self.assertEqual(f(2), 2) # lower semicontinuous
        self.assertIs(type(f(2)), float)
        self.assertEqual(f(2.5), 3.5)
        self.assertIs(type(f(2.5)), float)
        self.assertEqual(f(3), 4)
        self.assertIs(type(f(3)), float)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

        # another step function
        g = PiecewiseLinearFunction([1,1,2,3],
                                    [1,2,3,4],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3) # lower semicontinuous
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

        # another step function
        g = PiecewiseLinearFunction([1,2,3,3],
                                    [1,2,3,4],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3) # lower semicontinuous
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

        # another step function using parameters
        g = PiecewiseLinearFunction([pmo.parameter(1),
                                     pmo.parameter(1),
                                     pmo.parameter(2),
                                     pmo.parameter(3)],
                                    [pmo.parameter(1),
                                     pmo.parameter(2),
                                     pmo.parameter(3),
                                     pmo.parameter(4)],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3) # lower semicontinuous
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

        # another step function
        g = PiecewiseLinearFunction([1,1,2,3,4],
                                    [1,2,3,4,5],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1) # lower semicontinuous
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3)
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        self.assertEqual(f(3.5), 4.5)
        self.assertEqual(f(4), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)

        # another step function
        g = PiecewiseLinearFunction([1,2,2,3,4],
                                    [1,2,3,4,5],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2) # lower semicontinuous
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        self.assertEqual(f(3.5), 4.5)
        self.assertEqual(f(4), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)

        # another step function
        g = PiecewiseLinearFunction([1,2,3,3,4],
                                    [1,2,3,4,5],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3) # lower semicontinuous
        self.assertEqual(f(3.5), 4.5)
        self.assertEqual(f(4), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)

        # another step function
        g = PiecewiseLinearFunction([1,2,3,4,4],
                                    [1,2,3,4,5],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3)
        self.assertEqual(f(3.5), 3.5)
        self.assertEqual(f(4), 4) # lower semicontinuous
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)

        # another step function
        g = PiecewiseLinearFunction([1,2,3,4,5],
                                    [1,2,3,4,5],
                                    equal_slopes_tolerance=-1)
        f = TransformedPiecewiseLinearFunction(
            g,
            require_bounded_input_variable=False,
            equal_slopes_tolerance=-1)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 2.5)
        self.assertEqual(f(3), 3)
        self.assertEqual(f(3.5), 3.5)
        self.assertEqual(f(4), 4)
        self.assertEqual(f(4.5), 4.5)
        self.assertEqual(f(5), 5)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(5.1)

    def test_type(self):
        for key in transforms.registered_transforms:
            p = transforms.piecewise([1,2,3],
                                     [1,2,1],
                                     repn=key,
                                     validate=False)
            self.assertTrue(len(list(p.children())) <= 4)
            self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunction))
            self.assertTrue(isinstance(p, transforms.registered_transforms[key]))
            self.assertTrue(isinstance(p, ICategorizedObject))
            self.assertTrue(isinstance(p, ICategorizedObjectContainer))
            self.assertTrue(isinstance(p, IHeterogeneousContainer))
            self.assertTrue(isinstance(p, IBlock))
            self.assertTrue(isinstance(p, block))

    def test_bad_repn(self):
        repn = list(transforms.registered_transforms.keys())[0]
        self.assertTrue(repn in transforms.registered_transforms)
        transforms.piecewise([1,2,3],
                             [1,2,1],
                             validate=False,
                             repn=repn)

        repn = '_bad_repn_'
        self.assertFalse(repn in transforms.registered_transforms)
        with self.assertRaises(ValueError):
            transforms.piecewise([1,2,3],
                                 [1,2,1],
                                 validate=False,
                                 repn=repn)
        with self.assertRaises(ValueError):
            transforms.piecewise([1,2,3],
                                 [1,2,1],
                                 input=variable(lb=1,ub=3),
                                 validate=True,
                                 simplify=False,
                                 repn=repn)
        with self.assertRaises(ValueError):
            transforms.piecewise([1,2,3],
                                 [1,2,1],
                                 input=variable(lb=1,ub=3),
                                 validate=True,
                                 simplify=True,
                                 repn=repn)

    def test_init(self):
        for key in transforms.registered_transforms:
            for bound in ['lb','ub','eq','bad']:
                for args in [([1,2,3], [1,2,1]),
                             ([1,2,3,4,5],[1,2,1,2,1]),
                             ([1,2,3,4,5,6,7,8,9],[1,2,1,2,1,2,1,2,1])]:
                    kwds = {'repn': key, 'bound': bound, 'validate': False}
                    if bound == 'bad':
                        with self.assertRaises(ValueError):
                            transforms.piecewise(*args, **kwds)
                        kwds['simplify'] = True
                        with self.assertRaises(ValueError):
                            transforms.piecewise(*args, **kwds)
                        kwds['simplify'] = False
                        with self.assertRaises(ValueError):
                            transforms.piecewise(*args, **kwds)
                    else:
                        p = transforms.piecewise(*args, **kwds)
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms[key]))
                        self.assertTrue(
                            isinstance(p, TransformedPiecewiseLinearFunction))
                        self.assertEqual(p.active, True)
                        self.assertIs(p.parent, None)
                        kwds['simplify'] = True
                        p = transforms.piecewise(*args, **kwds)
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms[key]))
                        self.assertTrue(
                            isinstance(p, TransformedPiecewiseLinearFunction))
                        self.assertEqual(p.active, True)
                        self.assertIs(p.parent, None)
                        kwds['simplify'] = False
                        p = transforms.piecewise(*args, **kwds)
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms[key]))
                        self.assertTrue(
                            isinstance(p, TransformedPiecewiseLinearFunction))
                        self.assertEqual(p.active, True)
                        self.assertIs(p.parent, None)

    def test_bad_init(self):

        # lists not the same length
        with self.assertRaises(ValueError):
            PiecewiseLinearFunction([1,2,3],
                                    [1,2,1,1],
                                    validate=False)
        # lists not the same length
        with self.assertRaises(ValueError):
            PiecewiseLinearFunction([1,2,3,4],
                                    [1,2,1],
                                    validate=False)

        # breakpoints list not nondecreasing
        with self.assertRaises(util.PiecewiseValidationError):
            PiecewiseLinearFunction([1,3,2],
                                    [1,2,1])

        PiecewiseLinearFunction([1,3,2],
                                [1,2,1],
                                validate=False)

        PiecewiseLinearFunction([1,2,3],
                                [1,1,1+2e-6],
                                equal_slopes_tolerance=1e-6)

        # consecutive slopes are "equal"
        with self.assertRaises(util.PiecewiseValidationError):
            PiecewiseLinearFunction([1,2,3],
                                    [1,1,1+2e-6],
                                    equal_slopes_tolerance=3e-6)

        PiecewiseLinearFunction([1,2,3],
                                [1,1,1+2e-6],
                                validate=False)

        f = PiecewiseLinearFunction([1,2,3],
                                    [1,2,1])
        TransformedPiecewiseLinearFunction(f,
                                           input=variable(lb=1,ub=3),
                                           require_bounded_input_variable=True)

        TransformedPiecewiseLinearFunction(f,
                                input=variable(lb=1,ub=3),
                                require_bounded_input_variable=False)

        # variable is not bounded
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f,
                                    input=variable(lb=1),
                                    require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f,
                                input=variable(lb=1),
                                require_bounded_input_variable=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f,
                                    input=variable(ub=3),
                                    require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f,
                                input=variable(ub=3),
                                require_bounded_input_variable=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f,
                                    require_bounded_input_variable=True)
        TransformedPiecewiseLinearFunction(f,
                                require_bounded_input_variable=False)

        # variable domain is not fully covered
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f,
                                    input=variable(lb=0),
                                    require_bounded_input_variable=False,
                                    require_variable_domain_coverage=True)
        TransformedPiecewiseLinearFunction(f,
                                input=variable(lb=0),
                                require_bounded_input_variable=False,
                                require_variable_domain_coverage=False)
        with self.assertRaises(util.PiecewiseValidationError):
            TransformedPiecewiseLinearFunction(f,
                                    input=variable(ub=4),
                                    require_bounded_input_variable=False,
                                    require_variable_domain_coverage=True)
        TransformedPiecewiseLinearFunction(f,
                                input=variable(ub=4),
                                require_bounded_input_variable=False,
                                require_variable_domain_coverage=False)

    def test_bad_init_log_types(self):
        # lists are not of length: (2^n) + 1
        with self.assertRaises(ValueError):
            transforms.piecewise([1,2,3,4],[1,2,3,4],repn='dlog',validate=False)
        with self.assertRaises(ValueError):
            transforms.piecewise([1,2,3,4],[1,2,3,4],repn='log',validate=False)

    def test_step(self):
        breakpoints = [1,2,2]
        values = [1,0,1]
        v = variable()
        v.bounds = min(breakpoints), max(breakpoints)
        for key in transforms.registered_transforms:
            if key in ('mc','convex'):
                with self.assertRaises(util.PiecewiseValidationError):
                    transforms.piecewise(breakpoints,
                                         values,
                                         input=v,
                                         repn=key)
            else:
                p = transforms.piecewise(breakpoints,
                                         values,
                                         input=v,
                                         repn=key)
                self.assertEqual(p.validate(), 4)

    def test_simplify(self):
        v = variable(lb=1, ub=3)
        convex_breakpoints = [1,2,3]
        convex_values = [1,0,1]
        for key in transforms.registered_transforms:
            for bound in ('lb','ub','eq'):
                if (key == 'convex') and \
                   (bound != 'lb'):
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(convex_breakpoints,
                                             convex_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=False)
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(convex_breakpoints,
                                             convex_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=True)
                else:
                    p = transforms.piecewise(convex_breakpoints,
                                             convex_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=False)
                    self.assertTrue(
                        isinstance(p, transforms.registered_transforms[key]))
                    self.assertEqual(p.validate(), util.characterize_function.convex)
                    p = transforms.piecewise(convex_breakpoints,
                                             convex_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=True)
                    if bound == 'lb':
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms['convex']))
                    else:
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms[key]))

        concave_breakpoints = [1,2,3]
        concave_values = [-1,0,-1]
        for key in transforms.registered_transforms:
            for bound in ('lb','ub','eq'):
                if (key == 'convex') and \
                   (bound != 'ub'):
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(concave_breakpoints,
                                             concave_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=False)
                    with self.assertRaises(util.PiecewiseValidationError):
                        transforms.piecewise(concave_breakpoints,
                                             concave_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=True)
                else:
                    p = transforms.piecewise(concave_breakpoints,
                                             concave_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=False)
                    self.assertTrue(
                        isinstance(p, transforms.registered_transforms[key]))
                    self.assertEqual(p.validate(), util.characterize_function.concave)
                    p = transforms.piecewise(concave_breakpoints,
                                             concave_values,
                                             input=v,
                                             repn=key,
                                             bound=bound,
                                             simplify=True)
                    if bound == 'ub':
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms['convex']))
                    else:
                        self.assertTrue(
                            isinstance(p, transforms.registered_transforms[key]))

        affine_breakpoints = [1,3]
        affine_values = [1,3]
        for key in transforms.registered_transforms:
            for bound in ('lb','ub','eq'):
                p = transforms.piecewise(affine_breakpoints,
                                         affine_values,
                                         input=v,
                                         repn=key,
                                         bound=bound,
                                         simplify=False)
                self.assertTrue(
                    isinstance(p, transforms.registered_transforms[key]))
                self.assertEqual(p.validate(), util.characterize_function.affine)
                p = transforms.piecewise(affine_breakpoints,
                                         affine_values,
                                         input=v,
                                         repn=key,
                                         bound=bound,
                                         simplify=True)
                self.assertTrue(
                    isinstance(p, transforms.registered_transforms['convex']))

class Test_piecewise_dict(_TestActiveDictContainerBase,
                          unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: transforms.piecewise([1,2,3],
                                                       [1,2,1],
                                                       validate=False)

class Test_piecewise_list(_TestActiveListContainerBase,
                          unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: transforms.piecewise([1,2,3],
                                                       [1,2,1],
                                                       validate=False)

@unittest.skipUnless(util.numpy_available and util.scipy_available,
                     "Numpy or Scipy is not available")
class Test_piecewise_nd(unittest.TestCase):

    def test_pickle(self):
        for key in transforms_nd.registered_transforms:
            p = transforms_nd.piecewise_nd(_test_tri,
                                           _test_values,
                                           repn=key)
            self.assertEqual(p.parent, None)
            pup = pickle.loads(
                pickle.dumps(p))
            self.assertEqual(pup.parent, None)
            b = block()
            b.p = p
            self.assertIs(p.parent, b)
            bup = pickle.loads(
                pickle.dumps(b))
            pup = bup.p
            self.assertIs(pup.parent, bup)

    def test_call(self):

        #
        # 2d points
        #
        vlist = variable_list([variable(lb=0, ub=1),
                               variable(lb=0, ub=1)])
        tri = util.generate_delaunay(vlist, num=3)
        x, y = tri.points.T
        values = x*y
        g = PiecewiseLinearFunctionND(tri, values)
        f = TransformedPiecewiseLinearFunctionND(g)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertTrue(util.numpy.isclose(f(tri.points), values).all())
        self.assertAlmostEqual(f([0,0]), 0.0)
        self.assertAlmostEqual(f(util.numpy.array([0,0])), 0.0)
        self.assertAlmostEqual(f([1,1]), 1.0)
        self.assertAlmostEqual(f(util.numpy.array([1,1])), 1.0)

        #
        # 3d points
        #
        vlist = variable_list([variable(lb=0, ub=1),
                               variable(lb=0, ub=1),
                               variable(lb=0, ub=1)])
        tri = util.generate_delaunay(vlist, num=10)
        x, y, z = tri.points.T
        values = x*y*z
        g = PiecewiseLinearFunctionND(tri, values)
        f = TransformedPiecewiseLinearFunctionND(g)
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, IBlock)
        self.assertTrue(util.numpy.isclose(f(tri.points), values).all())
        self.assertAlmostEqual(f([0,0,0]), 0.0)
        self.assertAlmostEqual(f(util.numpy.array([0,0,0])), 0.0)
        self.assertAlmostEqual(f([1,1,1]), 1.0)
        self.assertAlmostEqual(f(util.numpy.array([1,1,1])), 1.0)

    def test_type(self):
        for key in transforms_nd.registered_transforms:
            p = transforms_nd.piecewise_nd(_test_tri,
                                           _test_values,
                                           repn=key)
            # small block storage
            self.assertTrue(len(list(p.children())) <= 4)
            self.assertTrue(isinstance(p, TransformedPiecewiseLinearFunctionND))
            self.assertTrue(isinstance(p, transforms_nd.registered_transforms[key]))
            self.assertTrue(isinstance(p, ICategorizedObject))
            self.assertTrue(isinstance(p, ICategorizedObjectContainer))
            self.assertTrue(isinstance(p, IHeterogeneousContainer))
            self.assertTrue(isinstance(p, IBlock))
            self.assertTrue(isinstance(p, block))

    def test_bad_repn(self):
        repn = list(transforms_nd.registered_transforms.keys())[0]
        self.assertTrue(repn in transforms_nd.registered_transforms)
        transforms_nd.piecewise_nd(_test_tri,
                                   _test_values,
                                   repn=repn)

        repn = '_bad_repn_'
        self.assertFalse(repn in transforms_nd.registered_transforms)
        with self.assertRaises(ValueError):
            transforms_nd.piecewise_nd(_test_tri,
                                       _test_values,
                                       repn=repn)

    def test_init(self):
        for key in transforms_nd.registered_transforms:
            for bound in ['lb','ub','eq','bad']:
                args = (_test_tri, _test_values)
                kwds = {'repn': key, 'bound': bound}
                if bound == 'bad':
                    with self.assertRaises(ValueError):
                        transforms_nd.piecewise_nd(*args, **kwds)
                else:
                    p = transforms_nd.piecewise_nd(*args, **kwds)
                    self.assertTrue(
                        isinstance(p, transforms_nd.registered_transforms[key]))
                    self.assertTrue(
                        isinstance(p, TransformedPiecewiseLinearFunctionND))
                    self.assertEqual(p.active, True)
                    self.assertIs(p.parent, None)

@unittest.skipUnless(util.numpy_available and util.scipy_available,
                     "Numpy or Scipy is not available")
class Test_piecewise_nd_dict(_TestActiveDictContainerBase,
                             unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: \
                     transforms_nd.piecewise_nd(_test_tri,
                                                _test_values)

@unittest.skipUnless(util.numpy_available and util.scipy_available,
                     "Numpy or Scipy is not available")
class Test_piecewise_nd_list(_TestActiveListContainerBase,
                             unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self:\
                     transforms_nd.piecewise_nd(_test_tri,
                                                _test_values)

if __name__ == "__main__":
    unittest.main()
