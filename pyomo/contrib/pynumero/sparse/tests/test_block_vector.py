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

from pyomo.contrib.pynumero.sparse.block_vector import BlockVector


class TestBlockVector(unittest.TestCase):

    def test_constructor(self):

        v = BlockVector(4)
        self.assertEqual(v.nblocks, 4)
        self.assertEqual(v.bshape, (4,))
        self.assertEqual(v.size, 0)

        v[0] = np.ones(2)
        v[1] = np.ones(4)
        self.assertEqual(v.size, 6)
        self.assertEqual(v.shape, (6,))
        v[0] = None
        self.assertEqual(v.size, 4)
        self.assertEqual(v.shape, (4,))
        self.assertEqual(v.ndim, 1)

        with self.assertRaises(Exception) as context:
            BlockVector('hola')

    def setUp(self):

        self.ones = BlockVector(3)
        self.list_sizes_ones = [2, 4, 3]
        for idx, s in enumerate(self.list_sizes_ones):
            self.ones[idx] = np.ones(s)

    def test_block_sizes(self):
        self.assertListEqual(self.ones.block_sizes().tolist(), self.list_sizes_ones)

    def test_dot(self):
        v1 = self.ones
        self.assertEqual(v1.dot(v1), v1.size)
        v2 = v1.clone(3.3)
        self.assertAlmostEqual(v1.dot(v2), v1.size*3.3)
        self.assertAlmostEqual(v2.dot(v1.flatten()), v1.size*3.3)
        with self.assertRaises(Exception) as context:
            v1.dot(1.0)

    def test_mean(self):
        flat_v = np.ones(self.ones.size)
        v = self.ones
        self.assertEqual(v.mean(), flat_v.mean())
        v = BlockVector(2)
        self.assertEqual(v.mean(), 0.0)

    def test_sum(self):
        self.assertEqual(self.ones.sum(), self.ones.size)
        v = BlockVector(2)
        v[0] = np.arange(5)
        v[1] = np.arange(9)
        self.assertEqual(v.sum(), 46)

    def test_all(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.ones(3)
        v[0] = a
        v[1] = b
        self.assertTrue(v.all())

        v = BlockVector(2)
        a = np.zeros(5)
        b = np.zeros(3)
        v[0] = a
        v[1] = b
        self.assertFalse(v.all())

    def test_any(self):

        v = BlockVector(2)
        a = np.zeros(5)
        b = np.ones(3)
        v[0] = a
        v[1] = b
        self.assertTrue(v.any())

        v = BlockVector(2)
        a = np.zeros(5)
        b = np.zeros(3)
        v[0] = a
        v[1] = b
        self.assertFalse(v.any())

    def test_argpartition(self):
        v = self.ones
        self.assertRaises(NotImplementedError, v.argpartition, 1)

    def test_argsort(self):
        v = self.ones
        self.assertRaises(NotImplementedError, v.argsort)

    def test_astype(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.astype(np.int, copy=False)

        vv = v.astype(np.int)
        self.assertEqual(vv.nblocks, v.nblocks)
        for blk in vv:
            self.assertEqual(blk.dtype, np.int)

    def test_byteswap(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            v.byteswap()

    def test_choose(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            v.choose(1)

    def test_clip(self):

        v = BlockVector(3)
        v2 = BlockVector(3)
        a = np.zeros(5)
        b = np.ones(3)*5.0
        c = np.ones(3)*10.0

        v[0] = a
        v[1] = b
        v[2] = c

        v2[0] = np.ones(5) * 4.0
        v2[1] = np.ones(3) * 5.0
        v2[2] = np.ones(3) * 9.0

        vv = v.clip(4.0, 9.0)
        self.assertEqual(vv.nblocks, v.nblocks)
        for bid, blk in enumerate(vv):
            self.assertTrue(np.allclose(blk, v2[bid]))

        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.clip(4.0, 9.0, out=v2)

    def test_compress(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.compress(1, out=1)

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        c = v.compress(v < 1)

        v2 = BlockVector(2)
        b = np.zeros(9)
        v2[0] = np.ones(0)
        v2[1] = b

        self.assertEqual(c.nblocks, v.nblocks)
        for bid, blk in enumerate(c):
            self.assertTrue(np.allclose(blk, v2[bid]))

        flags = v < 1
        c = v.compress(flags.flatten())
        self.assertEqual(c.nblocks, v.nblocks)
        for bid, blk in enumerate(c):
            self.assertTrue(np.allclose(blk, v2[bid]))

        with self.assertRaises(Exception) as context:
            v.compress(1.0)

    def test_nonzero(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        n = v.nonzero()

        v2 = BlockVector(2)
        v2[0] = np.arange(5)
        v2[1] = np.zeros(0)
        self.assertEqual(n[0].nblocks, v.nblocks)
        for bid, blk in enumerate(n[0]):
            self.assertTrue(np.allclose(blk, v2[bid]))

    def test_ptp(self):

        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        v[0] = a
        v[1] = b

        vv = np.arange(9)
        self.assertEqual(vv.ptp(), v.ptp())

    def test_round(self):

        v = BlockVector(2)
        a = np.ones(5)*1.1
        b = np.ones(9)*1.1
        v[0] = a
        v[1] = b

        vv = v.round()
        self.assertEqual(vv.nblocks, v.nblocks)
        a = np.ones(5)
        b = np.ones(9)
        v[0] = a
        v[1] = b
        for bid, blk in enumerate(vv):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_std(self):

        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        v[0] = a
        v[1] = b

        vv = np.concatenate([a, b])
        self.assertEqual(vv.std(), v.std())

    def test_conj(self):
        v = self.ones
        vv = v.conj()
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertEqual(vv.shape, v.shape)
        for bid, blk in enumerate(vv):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_conjugate(self):
        v = self.ones
        vv = v.conjugate()
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertEqual(vv.shape, v.shape)
        for bid, blk in enumerate(vv):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_diagonal(self):
        v = self.ones
        with self.assertRaises(ValueError) as ctx:
            vv = v.diagonal()

    def test_getfield(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.getfield(1)

    def test_item(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.item(1)

    def test_itemset(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.itemset(1)

    def test_newbyteorder(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.newbyteorder()

    def test_partition(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.partition(1)

    def test_repeat(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.repeat(1)

    def test_reshape(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.reshape(1)

    def test_resize(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.resize(1)

    def test_searchsorted(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.searchsorted(1)

    def test_setfield(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.setfield(1, 1)

    def test_setflags(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.setflags()

    def test_sort(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.sort()

    def test_squeeze(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.squeeze()

    def test_swapaxes(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.swapaxes(1, 1)

    def test_tobytes(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.tobytes()

    def test_trace(self):
        v = self.ones
        with self.assertRaises(NotImplementedError) as ctx:
            vv = v.trace()

    def test_prod(self):
        self.assertEqual(self.ones.prod(), 1)
        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        c = np.concatenate([a, b])
        v[0] = a
        v[1] = b
        self.assertEqual(v.prod(), c.prod())

    def test_max(self):
        self.assertEqual(self.ones.max(), 1)
        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        c = np.concatenate([a, b])
        v[0] = a
        v[1] = b
        self.assertEqual(v.max(), c.max())

    def test_min(self):
        self.assertEqual(self.ones.min(), 1)
        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        c = np.concatenate([a, b])
        v[0] = a
        v[1] = b
        self.assertEqual(v.min(), c.min())

    def test_tolist(self):
        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        c = np.concatenate([a, b])
        v[0] = a
        v[1] = b
        self.assertListEqual(v.tolist(), c.tolist())

    def test_flatten(self):
        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        c = np.concatenate([a, b])
        v[0] = a
        v[1] = b
        self.assertListEqual(v.flatten().tolist(), c.tolist())

    def test_fill(self):
        v = BlockVector(2)
        a = np.arange(5)
        b = np.arange(9)
        v[0] = a
        v[1] = b
        v.fill(1.0)
        c = np.ones(v.size)
        self.assertListEqual(v.tolist(), c.tolist())

    def test_shape(self):
        size = sum(self.list_sizes_ones)
        self.assertEqual(self.ones.shape, (size,))

    def test_bshape(self):
        self.assertEqual(self.ones.bshape, (3,))

    def test_size(self):
        size = sum(self.list_sizes_ones)
        self.assertEqual(self.ones.size, size)

    def test_argmax(self):
        v = BlockVector(2)
        v[0] = np.arange(5)
        v[1] = np.arange(10, 15)
        self.assertEqual(v.argmax(), v.size-1)

    def test_argmin(self):
        v = BlockVector(2)
        v[0] = np.arange(5)
        v[1] = np.arange(10, 15)
        self.assertEqual(v.argmin(), 0)

    def test_cumprod(self):

        v = BlockVector(3)
        v[0] = np.arange(1, 5)
        v[1] = np.arange(5, 10)
        v[2] = np.arange(10, 15)
        c = np.arange(1, 15)
        res = v.cumprod()
        self.assertIsInstance(res, BlockVector)
        self.assertEqual(v.nblocks, res.nblocks)
        self.assertTrue(np.allclose(c.cumprod(), res.flatten()))

    def test_cumsum(self):
        v = BlockVector(3)
        v[0] = np.arange(1, 5)
        v[1] = np.arange(5, 10)
        v[2] = np.arange(10, 15)
        c = np.arange(1, 15)
        res = v.cumsum()
        self.assertIsInstance(res, BlockVector)
        self.assertEqual(v.nblocks, res.nblocks)
        self.assertTrue(np.allclose(c.cumsum(), res.flatten()))

    def test_clone(self):
        v = self.ones
        w = v.clone()
        self.assertListEqual(w.tolist(), v.tolist())
        x = v.clone(4)
        self.assertListEqual(x.tolist(), [4]*v.size)
        y = x.clone(copy=False)
        y[2][-1] = 6
        d = np.ones(y.size)*4
        d[-1] = 6
        self.assertListEqual(y.tolist(), d.tolist())
        self.assertListEqual(x.tolist(), d.tolist())

    def test_add(self):

        v = self.ones
        v1 = self.ones
        result = v + v1
        self.assertListEqual(result.tolist(), [2]*v.size)
        result = v + 2
        self.assertListEqual(result.tolist(), [3] * v.size)
        result = v + v1.flatten()
        self.assertTrue(np.allclose(result.flatten(), v.flatten()+v1.flatten()))

        with self.assertRaises(Exception) as context:
            result = v + 'hola'

    def test_radd(self):
        v = self.ones
        v1 = self.ones
        result = v + v1
        self.assertListEqual(result.tolist(), [2] * v.size)
        result = 2 + v
        self.assertListEqual(result.tolist(), [3] * v.size)
        result = v.flatten() + v1
        self.assertTrue(np.allclose(result.flatten(), v.flatten() + v1.flatten()))

    def test_sub(self):
        v = self.ones
        v1 = self.ones
        result = v - v1
        self.assertListEqual(result.tolist(), [0] * v.size)
        result = v - 1.0
        self.assertListEqual(result.tolist(), [0] * v.size)
        result = v - v1.flatten()
        self.assertTrue(np.allclose(result.flatten(), v.flatten() - v1.flatten()))

        with self.assertRaises(Exception) as context:
            result = v - 'hola'

    def test_rsub(self):
        v = self.ones
        v1 = self.ones
        result = v1.__rsub__(v)
        self.assertListEqual(result.tolist(), [0] * v.size)
        result = 1.0 - v
        self.assertListEqual(result.tolist(), [0] * v.size)
        result = v.flatten() - v1
        self.assertTrue(np.allclose(result.flatten(), v.flatten() - v1.flatten()))

    def test_mul(self):
        v = self.ones
        v1 = v.clone(5, copy=True)
        result = v1 * v
        self.assertListEqual(result.tolist(), [5] * v.size)
        result = v * 5.0
        self.assertListEqual(result.tolist(), [5] * v.size)
        result = v * v1.flatten()
        self.assertTrue(np.allclose(result.flatten(), v.flatten() * v1.flatten()))

        with self.assertRaises(Exception) as context:
            result = v * 'hola'

    def test_rmul(self):
        v = self.ones
        v1 = v.clone(5, copy=True)
        result = v1.__rmul__(v)
        self.assertListEqual(result.tolist(), [5] * v.size)
        result = v * 5.0
        self.assertListEqual(result.tolist(), [5] * v.size)
        result = v.flatten() * v1
        self.assertTrue(np.allclose(result.flatten(), v.flatten() * v1.flatten()))

    # @unittest.skipIf(sys.version_info < (3, 0), 'not supported in this veresion')
    # def test_truediv(self):
    #     v = self.ones
    #     v1 = v.clone(5, copy=True)
    #     result = v / v1
    #     self.assertListEqual(result.tolist(), [1/5] * v.size)
    #     result = v / v1.flatten()
    #     self.assertTrue(np.allclose(result.flatten(), v.flatten() / v1.flatten()))
    #
    # @unittest.skipIf(sys.version_info < (3, 0), 'not supported in this veresion')
    # def test_rtruediv(self):
    #     v = self.ones
    #     v1 = v.clone(5, copy=True)
    #     result = v1.__rtruediv__(v)
    #     self.assertListEqual(result.tolist(), [1 / 5] * v.size)
    #     result = v.flatten() / v1
    #     self.assertTrue(np.allclose(result.flatten(), v.flatten() / v1.flatten()))
    #
    # def test_floordiv(self):
    #     v = self.ones
    #     v.fill(2)
    #     v1 = v.clone(5, copy=True)
    #     result = v1 // v
    #     self.assertListEqual(result.tolist(), [5 // 2] * v.size)
    #     result = v // v1.flatten()
    #     self.assertTrue(np.allclose(result.flatten(), v.flatten() // v1.flatten()))
    #
    # def test_rfloordiv(self):
    #     v = self.ones
    #     v.fill(2)
    #     v1 = v.clone(5, copy=True)
    #     result = v.__rfloordiv__(v1)
    #     self.assertListEqual(result.tolist(), [5 // 2] * v.size)
    #     result = v.flatten() // v1
    #     self.assertTrue(np.allclose(result.flatten(), v.flatten() // v1.flatten()))

    def test_iadd(self):
        v = self.ones
        v += 3
        self.assertListEqual(v.tolist(), [4]*v.size)
        v.fill(1.0)
        v += v
        self.assertListEqual(v.tolist(), [2] * v.size)
        v.fill(1.0)
        v += np.ones(v.size)*3
        self.assertTrue(np.allclose(v.flatten(), np.ones(v.size)*4))

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b
        v += 1.0

        self.assertTrue(np.allclose(v[0], a + 1))
        self.assertTrue(np.allclose(v[1], b + 1))

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        v2 = BlockVector(2)
        v2[0] = np.ones(5)
        v2[1] = np.ones(9)

        v += v2
        self.assertTrue(np.allclose(v[0], a + 1))
        self.assertTrue(np.allclose(v[1], b + 1))

        self.assertTrue(np.allclose(v2[0], np.ones(5)))
        self.assertTrue(np.allclose(v2[1], np.ones(9)))

        with self.assertRaises(Exception) as context:
            v += 'hola'

    def test_isub(self):
        v = self.ones
        v -= 3
        self.assertListEqual(v.tolist(), [-2] * v.size)
        v.fill(1.0)
        v -= v
        self.assertListEqual(v.tolist(), [0] * v.size)
        v.fill(1.0)
        v -= np.ones(v.size) * 3
        self.assertTrue(np.allclose(v.flatten(), -np.ones(v.size) * 2))

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b
        v -= 5.0

        self.assertTrue(np.allclose(v[0], a - 5.0))
        self.assertTrue(np.allclose(v[1], b - 5.0))

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        v2 = BlockVector(2)
        v2[0] = np.ones(5)
        v2[1] = np.ones(9)

        v -= v2
        self.assertTrue(np.allclose(v[0], a - 1))
        self.assertTrue(np.allclose(v[1], b - 1))

        self.assertTrue(np.allclose(v2[0], np.ones(5)))
        self.assertTrue(np.allclose(v2[1], np.ones(9)))

        with self.assertRaises(Exception) as context:
            v -= 'hola'

    def test_imul(self):
        v = self.ones
        v *= 3
        self.assertListEqual(v.tolist(), [3] * v.size)
        v.fill(1.0)
        v *= v
        self.assertListEqual(v.tolist(), [1] * v.size)
        v.fill(1.0)
        v *= np.ones(v.size) * 2
        self.assertTrue(np.allclose(v.flatten(), np.ones(v.size) * 2))

        v = BlockVector(2)
        a = np.ones(5)
        b = np.arange(9)
        v[0] = a
        v[1] = b
        v *= 2.0

        self.assertTrue(np.allclose(v[0], a * 2.0))
        self.assertTrue(np.allclose(v[1], b * 2.0))

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        v2 = BlockVector(2)
        v2[0] = np.ones(5) * 2
        v2[1] = np.ones(9) * 2

        v *= v2
        self.assertTrue(np.allclose(v[0], a * 2))
        self.assertTrue(np.allclose(v[1], b * 2))

        self.assertTrue(np.allclose(v2[0], np.ones(5) * 2))
        self.assertTrue(np.allclose(v2[1], np.ones(9) * 2))

        with self.assertRaises(Exception) as context:
            v *= 'hola'

    def test_getitem(self):
        v = self.ones
        for i, s in enumerate(self.list_sizes_ones):
            self.assertEqual(v[i].size, s)
            self.assertEqual(v[i].shape, (s,))
            self.assertListEqual(v[i].tolist(), np.ones(s).tolist())

    def test_setitem(self):
        v = self.ones
        for i, s in enumerate(self.list_sizes_ones):
            v[i] = np.ones(s) * i
        for i, s in enumerate(self.list_sizes_ones):
            self.assertEqual(v[i].size, s)
            self.assertEqual(v[i].shape, (s,))
            res = np.ones(s) * i
            self.assertListEqual(v[i].tolist(), res.tolist())

    def test_set_blocks(self):
        v = self.ones
        blocks = [np.ones(s)*i for i, s in enumerate(self.list_sizes_ones)]
        v.set_blocks(blocks)
        for i, s in enumerate(self.list_sizes_ones):
            self.assertEqual(v[i].size, s)
            self.assertEqual(v[i].shape, (s,))
            res = np.ones(s) * i
            self.assertListEqual(v[i].tolist(), res.tolist())

    def test_has_none(self):
        v = self.ones
        self.assertFalse(v.has_none)
        v[0] = None
        self.assertTrue(v.has_none)
        v[0] = np.ones(2)
        self.assertFalse(v.has_none)

    def test_copyfrom(self):
        v = self.ones
        v1 = np.zeros(v.size)
        v.copyfrom(v1)
        self.assertListEqual(v.tolist(), v1.tolist())

        v2 = BlockVector(len(self.list_sizes_ones))
        for i, s in enumerate(self.list_sizes_ones):
            v2[i] = np.ones(s)*i
        v.copyfrom(v2)
        for idx, blk in enumerate(v2):
            self.assertListEqual(blk.tolist(), v2[idx].tolist())

        v3 = BlockVector(2)
        v4 = v.clone(2)
        v3[0] = v4
        v3[1] = np.zeros(3)
        self.assertListEqual(v3.tolist(), v4.tolist() + [0]*3)

    def test_copyto(self):
        v = self.ones
        v2 = BlockVector(len(self.list_sizes_ones))
        v.copyto(v2)
        self.assertListEqual(v.tolist(), v2.tolist())
        v3 = np.zeros(v.size)
        v.copyto(v3)
        self.assertListEqual(v.tolist(), v3.tolist())
        v *= 5
        v.copyto(v2)
        self.assertListEqual(v.tolist(), v2.tolist())

    def test_gt(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        flags = v > 0
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        flags = v > np.zeros(v.size)
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        vv = v.copy()
        vv.fill(0.0)
        flags = v > vv
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_ge(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        flags = v >= 0
        v[1] = b + 1
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))
        v[1] = b - 1
        flags = v >= np.zeros(v.size)
        v[1] = b
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        v[1] = b - 1
        vv = v.copy()
        vv.fill(0.0)
        flags = v >= vv
        v[1] = b
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_lt(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        flags = v < 1
        v[0] = a-1
        v[1] = b+1
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))
        v[0] = a + 1
        v[1] = b - 1
        flags = v < np.ones(v.size)
        v[0] = a - 1
        v[1] = b + 1
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        v[0] = a + 1
        v[1] = b - 1
        vv = v.copy()
        vv.fill(1.0)
        flags = v < vv
        v[0] = a - 1
        v[1] = b + 1
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_le(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        flags = v <= 1
        v[1] = b + 1
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        flags = v <= v
        vv = v.copy()
        vv.fill(1.0)
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, vv[bid]))

        flags = v <= v.flatten()
        vv = v.copy()
        vv.fill(1.0)
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, vv[bid]))

    def test_eq(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        flags = v == 1
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        flags = v == np.ones(v.size)
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        vv = v.copy()
        vv.fill(1.0)
        flags = v == vv
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_ne(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        flags = v != 0
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        flags = v != np.zeros(v.size)
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

        vv = v.copy()
        vv.fill(0.0)
        flags = v != vv
        self.assertEqual(v.nblocks, flags.nblocks)
        for bid, blk in enumerate(flags):
            self.assertTrue(np.allclose(blk, v[bid]))

    def test_contains(self):

        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b

        self.assertTrue(0 in v)
        self.assertFalse(3 in v)
    # ToDo: add tests for block vectors with block vectors in them
    # ToDo: vector comparisons
    def test_copy(self):
        v = BlockVector(2)
        a = np.ones(5)
        b = np.zeros(9)
        v[0] = a
        v[1] = b
        v2 = v.copy()
        self.assertTrue(np.allclose(v.flatten(), v2.flatten()))

    def test_unary_ufuncs(self):

        v = BlockVector(2)
        a = np.ones(3) * 0.5
        b = np.ones(2) * 0.8
        v[0] = a
        v[1] = b

        v2 = BlockVector(2)

        unary_funcs = [np.log10, np.sin, np.cos, np.exp, np.ceil,
                       np.floor, np.tan, np.arctan, np.arcsin,
                       np.arccos, np.sinh, np.cosh, np.abs,
                       np.tanh, np.arcsinh, np.arctanh,
                       np.fabs, np.sqrt, np.log, np.log2,
                       np.absolute, np.isfinite, np.isinf, np.isnan,
                       np.log1p, np.logical_not, np.exp2, np.expm1,
                       np.sign, np.rint, np.square, np.positive,
                       np.negative, np.rad2deg, np.deg2rad,
                       np.conjugate, np.reciprocal]

        for fun in unary_funcs:
            v2[0] = fun(v[0])
            v2[1] = fun(v[1])
            res = fun(v)
            self.assertIsInstance(res, BlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in range(2):
                self.assertTrue(np.allclose(res[i], v2[i]))

        other_funcs = [np.cumsum, np.cumprod, np.cumproduct]

        for fun in other_funcs:
            res = fun(v)
            self.assertIsInstance(res, BlockVector)
            self.assertEqual(res.nblocks, 2)
            self.assertTrue(np.allclose(fun(v.flatten()), res.flatten()))

        with self.assertRaises(Exception) as context:
            np.cbrt(v)

    def test_reduce_ufuncs(self):

        v = BlockVector(2)
        a = np.ones(3) * 0.5
        b = np.ones(2) * 0.8
        v[0] = a
        v[1] = b

        reduce_funcs = [np.sum, np.max, np.min, np.prod, np.mean]
        for fun in reduce_funcs:
            self.assertAlmostEqual(fun(v), fun(v.flatten()))

        other_funcs = [np.all, np.any, np.std, np.ptp, np.argmax, np.argmin]
        for fun in other_funcs:
            self.assertAlmostEqual(fun(v), fun(v.flatten()))

    def test_binary_ufuncs(self):

        v = BlockVector(2)
        a = np.ones(3) * 0.5
        b = np.ones(2) * 0.8
        v[0] = a
        v[1] = b

        v2 = BlockVector(2)
        a2 = np.ones(3) * 3.0
        b2 = np.ones(2) * 2.8
        v2[0] = a2
        v2[1] = b2

        binary_ufuncs = [np.add, np.multiply, np.divide, np.subtract,
                         np.greater, np.greater_equal, np.less,
                         np.less_equal, np.not_equal,
                         np.maximum, np.minimum,
                         np.fmax, np.fmin, np.equal,
                         np.logaddexp, np.logaddexp2, np.remainder,
                         np.heaviside, np.hypot]

        for fun in binary_ufuncs:
            flat_res = fun(v.flatten(), v2.flatten())
            res = fun(v, v2)
            self.assertTrue(np.allclose(flat_res, res.flatten()))

            res = fun(v, v2.flatten())
            self.assertTrue(np.allclose(flat_res, res.flatten()))

            res = fun(v.flatten(), v2)
            self.assertTrue(np.allclose(flat_res, res.flatten()))

            flat_res = fun(v.flatten(), 5)
            res = fun(v, 5)
            self.assertTrue(np.allclose(flat_res, res.flatten()))

            flat_res = fun(3.0, v2.flatten())
            res = fun(3.0, v2)
            self.assertTrue(np.allclose(flat_res, res.flatten()))

        v = BlockVector(2)
        a = np.ones(3, dtype=bool)
        b = np.ones(2, dtype=bool)
        v[0] = a
        v[1] = b

        v2 = BlockVector(2)
        a2 = np.zeros(3, dtype=bool)
        b2 = np.zeros(2, dtype=bool)
        v2[0] = a2
        v2[1] = b2

        binary_ufuncs = [np.logical_and, np.logical_or, np.logical_xor]
        for fun in binary_ufuncs:
            flat_res = fun(v.flatten(), v2.flatten())
            res = fun(v, v2)
            self.assertTrue(np.allclose(flat_res, res.flatten()))

if __name__ == '__main__':
    unittest.main()
