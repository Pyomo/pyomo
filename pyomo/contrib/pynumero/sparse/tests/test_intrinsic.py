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

    def test_ceil(self):

        np_ceil = np.ceil(self.v1)
        pn_ceil = pn.ceil(self.v1)
        self.assertTrue(np.allclose(np_ceil, pn_ceil))

        np_ceil1 = np.ceil(self.v1)
        np_ceil2 = np.ceil(self.v2)
        pn_block_ceil = pn.ceil(self.bv)
        self.assertTrue(np.allclose(np_ceil1, pn_block_ceil[0]))
        self.assertTrue(np.allclose(np_ceil2, pn_block_ceil[1]))

    def test_floor(self):
        np_v1 = np.floor(self.v1)
        pn_v1 = pn.floor(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = np.floor(self.v1)
        np_v2 = np.floor(self.v2)
        pn_block = pn.floor(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_exp(self):
        np_v1 = np.exp(self.v1)
        pn_v1 = pn.exp(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = np.exp(self.v1)
        np_v2 = np.exp(self.v2)
        pn_block = pn.exp(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_conjugate(self):
        fname = 'conjugate'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_sin(self):
        fname = 'sin'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_cos(self):
        fname = 'cos'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_tan(self):
        fname = 'tan'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_arctan(self):
        fname = 'arctan'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_arcsinh(self):
        fname = 'arcsinh'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_sinh(self):
        fname = 'sinh'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_cosh(self):
        fname = 'cosh'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_abs(self):
        fname = 'abs'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_absolute(self):
        fname = 'absolute'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_fabs(self):
        fname = 'fabs'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_around(self):
        fname = 'around'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_sqrt(self):
        fname = 'sqrt'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_log(self):
        fname = 'log'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_log2(self):
        fname = 'log2'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_log10(self):
        fname = 'log10'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_arcsin(self):
        self.v1 = np.array([1.1, 2.2, 3.3])/10.0
        self.v2 = np.array([4.4, 5.5, 6.6, 7.7])/10.0
        self.bv = BlockVector([self.v1, self.v2])
        fname = 'arcsin'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_arccos(self):
        self.v1 = np.array([1.1, 2.2, 3.3]) / 10.0
        self.v2 = np.array([4.4, 5.5, 6.6, 7.7]) / 10.0
        self.bv = BlockVector([self.v1, self.v2])
        fname = 'arccos'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_arccosh(self):
        fname = 'arccosh'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertTrue(np.allclose(np_v1, pn_v1))

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertTrue(np.allclose(np_v1, pn_block[0]))
        self.assertTrue(np.allclose(np_v2, pn_block[1]))

    def test_sum(self):
        fname = 'sum'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertEqual(np_v1, pn_v1)

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertAlmostEqual(np_v1 + np_v2, pn_block)

    def test_min(self):
        fname = 'min'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertEqual(np_v1, pn_v1)

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertAlmostEqual(min(np_v1, np_v2), pn_block)

    def test_max(self):
        fname = 'max'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertEqual(np_v1, pn_v1)

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertAlmostEqual(max(np_v1, np_v2), pn_block)

    def test_mean(self):
        fname = 'mean'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertEqual(np_v1, pn_v1)

        np_v1 = fnp(np.concatenate([self.v1, self.v2]))
        pn_block = fpn(self.bv)
        self.assertAlmostEqual(np_v1, pn_block)

    def test_prod(self):
        fname = 'prod'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v1 = fnp(self.v1)
        pn_v1 = fpn(self.v1)
        self.assertEqual(np_v1, pn_v1)

        np_v1 = fnp(self.v1)
        np_v2 = fnp(self.v2)
        pn_block = fpn(self.bv)
        self.assertAlmostEqual(np_v1 * np_v2, pn_block)

    def test_add(self):
        fname = 'add'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v = fnp(self.v1, self.v3)
        pn_v = fpn(self.v1, self.v3)
        self.assertTrue(np.allclose(np_v, pn_v))

        v = np.concatenate([self.v1, self.v2])
        bv = self.bv
        pn_v = fpn(v, bv)
        self.assertTrue(np.allclose(v*2, pn_v))
        pn_v = fpn(bv, v)
        self.assertTrue(np.allclose(v * 2, pn_v))

        v = np.concatenate([self.v1, self.v2])
        v2 = np.concatenate([self.v3, self.v4])
        pn_v = fpn(self.bv, self.bv2)
        self.assertTrue(np.allclose(fnp(v, v2), pn_v))

        with self.assertRaises(Exception) as context:
            fpn(self.bv.tolist(), self.bv2)

    def test_subtract(self):
        fname = 'subtract'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v = fnp(self.v1, self.v3)
        pn_v = fpn(self.v1, self.v3)
        self.assertTrue(np.allclose(np_v, pn_v))

        v = np.concatenate([self.v1, self.v2])
        bv = self.bv
        pn_v = fpn(v, bv)
        self.assertTrue(np.allclose(np.zeros(len(v)), pn_v))
        pn_v = fpn(bv, v)
        self.assertTrue(np.allclose(np.zeros(len(v)), pn_v))

        v = np.concatenate([self.v1, self.v2])
        v2 = np.concatenate([self.v3, self.v4])
        pn_v = fpn(self.bv, self.bv2)
        self.assertTrue(np.allclose(fnp(v, v2), pn_v))

        with self.assertRaises(Exception) as context:
            fpn(self.bv.tolist(), self.bv2)

    def test_multiply(self):
        fname = 'multiply'
        fnp = getattr(np, fname)
        fpn = getattr(pn, fname)
        np_v = fnp(self.v1, self.v3)
        pn_v = fpn(self.v1, self.v3)
        self.assertTrue(np.allclose(np_v, pn_v))

        v = np.concatenate([self.v1, self.v2])
        bv = self.bv
        pn_v = fpn(v, bv)
        self.assertTrue(np.allclose(fnp(v, v), pn_v))
        pn_v = fpn(bv, v)
        self.assertTrue(np.allclose(fnp(v, v), pn_v))

        v = np.concatenate([self.v1, self.v2])
        v2 = np.concatenate([self.v3, self.v4])
        pn_v = fpn(self.bv, self.bv2)
        self.assertTrue(np.allclose(fnp(v, v2), pn_v))

        with self.assertRaises(Exception) as context:
            fpn(self.bv.tolist(), self.bv2)
