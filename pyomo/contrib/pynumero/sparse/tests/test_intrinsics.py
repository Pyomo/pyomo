#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import sys
import pyutilib.th as unittest
try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Pynumero needs scipy and numpy to run NLP tests")

from pyomo.contrib.pynumero.sparse import BlockVector
import pyomo.contrib.pynumero as pn


class TestSparseIntrinsics(unittest.TestCase):

    def setUp(self):
        self.v1 = np.array([1.1, 2.2, 3.3])
        self.v2 = np.array([4.4, 5.5, 6.6, 7.7])
        self.v3 = np.array([1.1, 2.2, 3.3])*2
        self.v4 = np.array([4.4, 5.5, 6.6, 7.7])*2
        self.bv = BlockVector([self.v1, self.v2])
        self.bv2 = BlockVector([self.v3, self.v4])

    def test_where(self):

        bv = self.bv
        condition = bv >= 4.5
        res = pn.where(condition)[0]
        for bid, blk in enumerate(res):
            self.assertTrue(np.allclose(blk, pn.where(bv[bid] >= 4.5)))

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

        bones = BlockVector([np.ones(3), np.ones(4)])

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
        test_bv[0] = a
        test_bv[1] = b

        res = pn.isin(bv, test_bv)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res[bid].size)
            res_flat = np.isin(blk, test_bv[bid])
            self.assertTrue(np.allclose(res[bid], res_flat))

        c = np.concatenate([a, b])
        res = pn.isin(bv, c)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res[bid].size)
            res_flat = np.isin(blk, c)
            self.assertTrue(np.allclose(res[bid], res_flat))

        res = pn.isin(bv, test_bv, invert=True)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res[bid].size)
            res_flat = np.isin(blk, test_bv[bid], invert=True)
            self.assertTrue(np.allclose(res[bid], res_flat))

        c = np.concatenate([a, b])
        res = pn.isin(bv, c, invert=True)
        for bid, blk in enumerate(bv):
            self.assertEqual(blk.size, res[bid].size)
            res_flat = np.isin(blk, c, invert=True)
            self.assertTrue(np.allclose(res[bid], res_flat))

    # ToDo: try np.copy on a blockvector

