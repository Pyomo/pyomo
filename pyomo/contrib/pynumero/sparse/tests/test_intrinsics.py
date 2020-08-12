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

from pyomo.contrib.pynumero.dependencies import (
    numpy as np, numpy_available, scipy_available
)
if not (numpy_available and scipy_available):
    raise unittest.SkipTest(
        "Pynumero needs scipy and numpy to run Sparse intrinsict tests")

from pyomo.contrib.pynumero.sparse import BlockVector
import pyomo.contrib.pynumero as pn


class TestSparseIntrinsics(unittest.TestCase):

    def setUp(self):
        self.v1 = np.array([1.1, 2.2, 3.3])
        self.v2 = np.array([4.4, 5.5, 6.6, 7.7])
        self.v3 = np.array([1.1, 2.2, 3.3])*2
        self.v4 = np.array([4.4, 5.5, 6.6, 7.7])*2
        self.bv = BlockVector(2)
        self.bv2 = BlockVector(2)
        self.bv.set_blocks([self.v1, self.v2])
        self.bv2.set_blocks([self.v3, self.v4])

    def test_where(self):

        bv = self.bv
        condition = bv >= 4.5
        res = pn.where(condition)[0]
        for bid, blk in enumerate(res):
            self.assertTrue(np.allclose(blk, pn.where(bv.get_block(bid) >= 4.5)))

        flat_condition = condition.flatten()
        res = pn.where(condition, 2.0, 1.0)
        res_flat = pn.where(flat_condition, 2.0, 1.0)
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        res = pn.where(condition, 2.0, np.ones(bv.size))
        res_flat = pn.where(flat_condition, 2.0, np.ones(bv.size))
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        res = pn.where(condition, np.ones(bv.size) * 2.0, 1.0)
        res_flat = pn.where(flat_condition, np.ones(bv.size) * 2.0, 1.0)
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        res = pn.where(condition, np.ones(bv.size) * 2.0, np.ones(bv.size))
        res_flat = pn.where(flat_condition, np.ones(bv.size) * 2.0, np.ones(bv.size))
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        bones = BlockVector(2)
        bones.set_blocks([np.ones(3), np.ones(4)])

        res = pn.where(condition, bones * 2.0, 1.0)
        res_flat = pn.where(flat_condition, np.ones(bv.size) * 2.0, 1.0)
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        res = pn.where(condition, 2.0, bones)
        res_flat = pn.where(flat_condition, 2.0, bones)
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        res = pn.where(condition, np.ones(bv.size) * 2.0, bones)
        res_flat = pn.where(flat_condition, np.ones(bv.size) * 2.0, np.ones(bv.size))
        self.assertTrue(np.allclose(res.flatten(), res_flat))

        res = pn.where(condition, bones * 2.0, np.ones(bv.size))
        res_flat = pn.where(flat_condition, np.ones(bv.size) * 2.0, np.ones(bv.size))
        self.assertTrue(np.allclose(res.flatten(), res_flat))

    def test_isin(self):

        bv = self.bv
        test_bv = BlockVector(2)
        a = np.array([1.1, 3.3])
        b = np.array([5.5, 7.7])
        test_bv.set_block(0, a)
        test_bv.set_block(1, b)

        res = pn.isin(bv, test_bv)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res.get_block(bid).size)
            res_flat = np.isin(blk, test_bv.get_block(bid))
            self.assertTrue(np.allclose(res.get_block(bid), res_flat))

        c = np.concatenate([a, b])
        res = pn.isin(bv, c)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res.get_block(bid).size)
            res_flat = np.isin(blk, c)
            self.assertTrue(np.allclose(res.get_block(bid), res_flat))

        res = pn.isin(bv, test_bv, invert=True)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res.get_block(bid).size)
            res_flat = np.isin(blk, test_bv.get_block(bid), invert=True)
            self.assertTrue(np.allclose(res.get_block(bid), res_flat))

        c = np.concatenate([a, b])
        res = pn.isin(bv, c, invert=True)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res.get_block(bid).size)
            res_flat = np.isin(blk, c, invert=True)
            self.assertTrue(np.allclose(res.get_block(bid), res_flat))

    # ToDo: try np.copy on a blockvector

    def test_intersect1d(self):

        vv1 = np.array([1.1, 3.3])
        vv2 = np.array([4.4, 7.7])
        bvv = BlockVector(2)
        bvv.set_blocks([vv1, vv2])
        res = pn.intersect1d(self.bv, bvv)
        self.assertIsInstance(res, BlockVector)
        self.assertTrue(np.allclose(res.get_block(0), vv1))
        self.assertTrue(np.allclose(res.get_block(1), vv2))
        vv3 = np.array([1.1, 7.7])
        res = pn.intersect1d(self.bv, vv3)
        self.assertIsInstance(res, BlockVector)
        self.assertTrue(np.allclose(res.get_block(0), np.array([1.1])))
        self.assertTrue(np.allclose(res.get_block(1), np.array([7.7])))
        res = pn.intersect1d(vv3, self.bv)
        self.assertIsInstance(res, BlockVector)
        self.assertTrue(np.allclose(res.get_block(0), np.array([1.1])))
        self.assertTrue(np.allclose(res.get_block(1), np.array([7.7])))

    def test_setdiff1d(self):

        vv1 = np.array([1.1, 3.3])
        vv2 = np.array([4.4, 7.7])
        bvv = BlockVector(2)
        bvv.set_blocks([vv1, vv2])
        res = pn.setdiff1d(self.bv, bvv)
        self.assertIsInstance(res, BlockVector)
        self.assertTrue(np.allclose(res.get_block(0), np.array([2.2])))
        self.assertTrue(np.allclose(res.get_block(1), np.array([5.5, 6.6])))
        vv3 = np.array([1.1, 7.7])
        res = pn.setdiff1d(self.bv, vv3)
        self.assertIsInstance(res, BlockVector)
        self.assertTrue(np.allclose(res.get_block(0), np.array([2.2, 3.3])))
        self.assertTrue(np.allclose(res.get_block(1), np.array([4.4, 5.5, 6.6])))
