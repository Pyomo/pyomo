import pickle

import pyutilib.th as unittest
from pyomo.core.base.component_interface import \
    (ICategorizedObject,
     IActiveObject,
     IComponent,
     IComponentContainer,
     _IActiveComponentContainer)
from pyomo.core.tests.unit.test_component_dict import \
    _TestActiveComponentDictBase
from pyomo.core.tests.unit.test_component_list import \
    _TestActiveComponentListBase
from pyomo.core.base.component_block import (IBlockStorage,
                                             block,
                                             block_dict,
                                             block_list,
                                             StaticBlock)
from pyomo.core.base.component_variable import variable
from pyomo.core.base.component_piecewise.transforms import \
    (registered_transforms,
     _PiecewiseLinearFunction,
     piecewise,
     piecewise_sos2,
     piecewise_dcc,
     piecewise_cc,
     piecewise_mc,
     piecewise_inc)
import pyomo.core.base.component_piecewise.util as util
from pyomo.core.base.block import Block

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

    def test_is_postive_power_of_two(self):
        self.assertEqual(util.is_postive_power_of_two(-8), False)
        self.assertEqual(util.is_postive_power_of_two(-4), False)
        self.assertEqual(util.is_postive_power_of_two(-3), False)
        self.assertEqual(util.is_postive_power_of_two(-2), False)
        self.assertEqual(util.is_postive_power_of_two(-1), False)
        self.assertEqual(util.is_postive_power_of_two(0), False)
        self.assertEqual(util.is_postive_power_of_two(1), True)
        self.assertEqual(util.is_postive_power_of_two(2), True)
        self.assertEqual(util.is_postive_power_of_two(3), False)
        self.assertEqual(util.is_postive_power_of_two(4), True)
        self.assertEqual(util.is_postive_power_of_two(5), False)
        self.assertEqual(util.is_postive_power_of_two(6), False)
        self.assertEqual(util.is_postive_power_of_two(7), False)
        self.assertEqual(util.is_postive_power_of_two(8), True)
        self.assertEqual(util.is_postive_power_of_two(15), False)
        self.assertEqual(util.is_postive_power_of_two(16), True)
        self.assertEqual(util.is_postive_power_of_two(31), False)
        self.assertEqual(util.is_postive_power_of_two(32), True)

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
        self.assertEqual(fc, 0) # affine
        self.assertEqual(slopes, [0,0])

        fc, slopes = util.characterize_function([1,2,3],
                                                [1,0,1])
        self.assertEqual(fc, 1) # convex
        self.assertEqual(slopes, [-1,1])

        fc, slopes = util.characterize_function([1,2,3],
                                                [1,2,1])
        self.assertEqual(fc, 2) # concave
        self.assertEqual(slopes, [1,-1])

        fc, slopes = util.characterize_function([1,1,2],
                                                [1,2,1])
        self.assertEqual(fc, 3) # step
        self.assertEqual(slopes, [None,-1])

        fc, slopes = util.characterize_function([1,2,3,4],
                                                [1,2,1,2])
        self.assertEqual(fc, 4) # none of the above
        self.assertEqual(slopes, [1,-1,1])

class Test_piecewise(unittest.TestCase):

    def test_pickle(self):
        for key in registered_transforms:
            p = piecewise([1,2,3],
                          [1,2,1],
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
        with self.assertRaises(ValueError):
            _PiecewiseLinearFunction([1,2,3],
                                     [1,2,1,1])
        with self.assertRaises(ValueError):
            _PiecewiseLinearFunction([1,2,3,4],
                                     [1,2,1])

        f = _PiecewiseLinearFunction([1,2,3],
                                     [1,2,1])
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, Block)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(2.5), 1.5)
        self.assertEqual(f(3), 1)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(4.1)

        # step function
        f = _PiecewiseLinearFunction([1,2,2,3],
                                     [1,2,3,4])
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, Block)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 1.5)
        self.assertEqual(f(2), 2) # lower semicontinuous
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

        # another step function
        f = _PiecewiseLinearFunction([1,1,2,3],
                                     [1,2,3,4])
        self.assertTrue(f.parent is None)
        self.assertEqual(f.ctype, Block)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(1.5), 2.5)
        self.assertEqual(f(2), 3) # lower semicontinuous
        self.assertEqual(f(2.5), 3.5)
        self.assertEqual(f(3), 4)
        with self.assertRaises(ValueError):
            f(0.9)
        with self.assertRaises(ValueError):
            f(3.1)

    def test_type(self):
        for key in registered_transforms:
            p = piecewise([1,2,3],
                          [1,2,1],
                          repn=key)
            self.assertTrue(isinstance(p, registered_transforms[key]))
            self.assertTrue(isinstance(p, ICategorizedObject))
            self.assertTrue(isinstance(p, IActiveObject))
            self.assertTrue(isinstance(p, IComponent))
            self.assertTrue(isinstance(p, IComponentContainer))
            self.assertTrue(isinstance(p, _IActiveComponentContainer))
            self.assertTrue(isinstance(p, StaticBlock))
            self.assertTrue(isinstance(p, IBlockStorage))

    def test_bad_repn(self):
        repn = 'sos2'
        self.assertTrue(repn in registered_transforms)
        piecewise([1,2,3],
                  [1,2,1],
                  repn=repn)

        repn = '_bad_repn_'
        self.assertFalse(repn in registered_transforms)
        with self.assertRaises(ValueError):
            piecewise([1,2,3],
                      [1,2,1],
                      repn=repn)

    def test_init(self):
        for key in registered_transforms:
            for bound in ['lb','ub','eq','bad']:
                args = ([1,2,3], [1,2,1])
                kwds = {'repn': key, 'bound': bound}
                if bound == 'bad':
                    with self.assertRaises(ValueError):
                        piecewise(*args, **kwds)
                else:
                    p = piecewise(*args, **kwds)
                    self.assertTrue(isinstance(p, registered_transforms[key]))
                    self.assertTrue(isinstance(p, _PiecewiseLinearFunction))
                    self.assertEqual(p.active, True)
                    self.assertIs(p.parent, None)

class Test_piecewise_dict(_TestActiveComponentDictBase,
                          unittest.TestCase):
    _container_type = block_dict
    _ctype_factory = lambda self: piecewise([1,2,3],
                                            [1,2,1])

class Test_piecewise_list(_TestActiveComponentListBase,
                          unittest.TestCase):
    _container_type = block_list
    _ctype_factory = lambda self: piecewise([1,2,3],
                                            [1,2,1])

if __name__ == "__main__":
    unittest.main()
