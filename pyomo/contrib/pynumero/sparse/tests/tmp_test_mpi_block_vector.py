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
try:
    from scipy.sparse import coo_matrix, bmat
    import numpy as np
except ImportError:
    raise unittest.SkipTest(
        "Pynumero needs scipy and numpy to run block matrix tests")

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    raise unittest.SkipTest(
        "Pynumero needs mpi4py to run mpi block vector tests")

try:
    from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
except ImportError:
    raise unittest.SkipTest(
        "Pynumero needs mpi4py to run mpi block vector tests")
try:
    from pyomo.contrib.pynumero.sparse import BlockVector
except ImportError:
    raise unittest.SkipTest(
        "Could not import BlockVector")

@unittest.skipIf(comm.Get_size() < 3, "Need at least 3 processors to run tests")
class TestMPIBlockVector(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # test problem 1

        v1 = MPIBlockVector(4, [0,1,0,1], comm)

        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3)
            v1[2] = np.ones(3)
        if rank == 1:
            v1[1] = np.zeros(2)
            v1[3] = np.ones(2)

        cls.v1 = v1

        v2 = MPIBlockVector(7, [0,0,1,1,2,2,-1], comm)

        rank = comm.Get_rank()
        if rank == 0:
            v2[0] = np.ones(2)
            v2[1] = np.ones(2)
        if rank == 1:
            v2[2] = np.zeros(3)
            v2[3] = np.zeros(3)
        if rank == 2:
            v2[4] = np.ones(4) * 2.0
            v2[5] = np.ones(4) * 2.0
        v2[6] = np.ones(2) * 3

        cls.v2 = v2

    def test_nblocks(self):
        v1 = self.v1
        self.assertEqual(v1.nblocks, 4)
        v2 = self.v2
        self.assertEqual(v2.nblocks, 7)

    def test_bshape(self):
        v1 = self.v1
        self.assertEqual(v1.bshape[0], 4)
        v2 = self.v2
        self.assertEqual(v2.bshape[0], 7)

    def test_size(self):
        v1 = self.v1
        self.assertEqual(v1.size, 10)
        v2 = self.v2
        self.assertEqual(v2.size, 20)

    def test_bshape(self):
        v1 = self.v1
        self.assertEqual(v1.shape[0], 10)
        v2 = self.v2
        self.assertEqual(v2.shape[0], 20)

    def test_ndim(self):
        v1 = self.v1
        self.assertEqual(v1.ndim, 1)

    def test_has_none(self):
        v = MPIBlockVector(4, [0,1,0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3)
            v[2] = np.ones(3)
        self.assertTrue(v.has_none)
        self.assertFalse(self.v1.has_none)

    def test_any(self):
        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3)
        if rank == 1:
            v[1] = np.zeros(3)
        self.assertTrue(v.any())
        if rank == 0:
            v[0] = None
        self.assertFalse(v.any())
        self.assertTrue(self.v1.any())
        self.assertTrue(self.v2.any())

    def test_all(self):
        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3)
        if rank == 1:
            v[1] = np.zeros(3)
        self.assertFalse(v.all())
        if rank == 1:
            v[1] = np.ones(3)
        self.assertTrue(v.all())
        if rank == 1:
            v[1] = None
        self.assertFalse(v.all())
        self.assertFalse(self.v1.all())
        self.assertFalse(self.v2.all())

    def test_min(self):
        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 10
        if rank == 1:
            v[1] = np.arange(3)
        self.assertEqual(v.min(), 0.0)
        if rank == 1:
            v[1] = -np.arange(3)
        self.assertEqual(v.min(), -2.0)

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 10
        if rank == 1:
            v[1] = np.arange(3)
        v[2] = -np.arange(6)
        self.assertEqual(v.min(), -5.0)
        self.assertEqual(self.v1.min(), 0.0)
        self.assertEqual(self.v2.min(), 0.0)

    def test_max(self):
        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 10
        if rank == 1:
            v[1] = np.arange(3)
        self.assertEqual(v.max(), 12.0)

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 10
        if rank == 1:
            v[1] = np.arange(3)
        v[2] = np.arange(60)
        self.assertEqual(v.max(), 59.0)
        self.assertEqual(self.v1.max(), 1.0)
        self.assertEqual(self.v2.max(), 3.0)

    def test_sum(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(3) + 3
        v[2] = np.arange(3) + 6

        b = np.arange(9)
        self.assertEqual(b.sum(), v.sum())
        self.assertEqual(self.v1.sum(), 8)
        self.assertEqual(self.v2.sum(), 26)

    def test_prod(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(2)
        if rank == 1:
            v[1] = np.ones(3)
        v[2] = np.ones(3)
        self.assertEqual(1.0, v.prod())
        if rank == 1:
            v[1] = np.ones(3) * 2
        self.assertEqual(8.0, v.prod())
        if rank == 0:
            v[0] = np.ones(2) * 3
        self.assertEqual(72.0, v.prod())
        self.assertEqual(0.0, self.v1.prod())
        self.assertEqual(0.0, self.v2.prod())

    def test_conj(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(2)
        if rank == 1:
            v[1] = np.ones(3)
        v[2] = np.ones(3)
        res = v.conj()
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        for j in v._owned_blocks:
            self.assertTrue(np.allclose(res[j], v[j].conj()))

    def test_conjugate(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(2)
        if rank == 1:
            v[1] = np.ones(3)
        v[2] = np.ones(3)
        res = v.conjugate()
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        for j in v._owned_blocks:
            self.assertTrue(np.allclose(res[j], v[j].conjugate()))

    def test_nonzero(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.array([0,1,2])
        if rank == 1:
            v[1] = np.array([0,0,2])
        v[2] = np.ones(3)
        res = v.nonzero()
        res = res[0]
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(res[0], np.array([1,2])))
        if rank == 1:
            self.assertTrue(np.allclose(res[1], np.array([2])))
        self.assertTrue(np.allclose(res[2], np.arange(3)))

        res = self.v1.nonzero()
        res = res[0]
        if rank == 0:
            self.assertTrue(np.allclose(res[0], np.arange(3)))
            self.assertTrue(np.allclose(res[2], np.arange(3)))
        if rank == 1:
            self.assertTrue(np.allclose(res[1], np.arange(0)))
            self.assertTrue(np.allclose(res[3], np.arange(2)))

    def test_round(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 0.01
        if rank == 1:
            v[1] = np.arange(3) + 3 + 0.01
        v[2] = np.arange(3) + 6 + 0.01

        res = v.round()
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(3)+3, res[1]))
        self.assertTrue(np.allclose(np.arange(3)+6, res[2]))

    def test_clip(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(3) + 3
        v[2] = np.arange(3) + 6

        res = v.clip(min=2.0)
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.array([2,2,2]), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(3)+3, res[1]))
        self.assertTrue(np.allclose(np.arange(3)+6, res[2]))

        res = v.clip(min=2.0, max=5.0)
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.array([2,2,2]), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.array([3,4,5]), res[1]))
        self.assertTrue(np.allclose(np.array([5,5,5]), res[2]))

        v1 = self.v1
        res = v1.clip(max=0.5)
        if rank == 0:
            self.assertTrue(np.allclose(np.ones(3) * 0.5, res[0]))
            self.assertTrue(np.allclose(np.ones(3) * 0.5, res[2]))
        if rank == 1:
            self.assertTrue(np.allclose(np.zeros(2), res[1]))
            self.assertTrue(np.allclose(np.ones(2) * 0.5, res[3]))

    def test_compress(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        cond = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            cond[0] = np.array([False, False, True])
        if rank == 1:
            cond[1] = np.array([True, True, True, False])
        cond[2] = np.array([True, True])

        res = v.compress(cond)
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.array([2]), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.array([0,1,2]), res[1]))
        self.assertTrue(np.allclose(np.array([0, 1]), res[2]))

        cond = BlockVector(3)
        cond[0] = np.array([False, False, True])
        cond[1] = np.array([True, True, True, False])
        cond[2] = np.array([True, True])

        res = v.compress(cond)
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(res.nblocks, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.array([2]), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.array([0,1,2]), res[1]))
        self.assertTrue(np.allclose(np.array([0, 1]), res[2]))

        with self.assertRaises(Exception) as context:
            res = v.compress(cond.flatten())

    def test_set_blocks(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        blocks = [np.arange(3), np.arange(4), np.arange(2)]
        v.set_blocks(blocks)
        rank = comm.Get_rank()
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4), v[1]))
        self.assertTrue(np.allclose(np.arange(2), v[2]))

    def test_owned_blocks(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        owned = v.owned_blocks
        rank = comm.Get_rank()
        if rank == 0:
            self.assertTrue(np.allclose(np.array([0, 2]), owned))
        if rank == 1:
            self.assertTrue(np.allclose(np.array([1, 2]), owned))

        owned = self.v1.owned_blocks
        if rank == 0:
            self.assertTrue(np.allclose(np.array([0, 2]), owned))
        if rank == 1:
            self.assertTrue(np.allclose(np.array([1, 3]), owned))

    def test_shared_blocks(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        shared = v.shared_blocks
        self.assertTrue(np.allclose(np.array([2]), shared))

    def test_clone(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        vv = v.clone()
        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))

    def test_copy(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        vv = v.copy()
        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))

    def test_copyto(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        vv = MPIBlockVector(3, [0,1,-1], comm)
        v.copyto(vv)

        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))

    def test_fill(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        v.fill(7.0)
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        self.assertTrue(np.allclose(np.array([2]), v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(np.ones(3)*7.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.ones(4)*7.0, v[1]))
        self.assertTrue(np.allclose(np.ones(2)*7.0, v[2]))

    def test_dot(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        all_v = np.concatenate([np.arange(3), np.arange(4), np.arange(2)])

        self.assertEqual(all_v.dot(all_v), v.dot(v))
        vv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])
        self.assertEqual(all_v.dot(all_v), v.dot(vv))

        with self.assertRaises(Exception) as context:
            v.dot(vv.flatten())

    def test_add(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        res = v + v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3)*2, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4)*2, res[1]))
        self.assertTrue(np.allclose(np.arange(2)*2, res[2]))

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])

        res = v + bv
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3)*2, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4)*2, res[1]))
        self.assertTrue(np.allclose(np.arange(2)*2, res[2]))

        res = bv + v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3)*2, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4)*2, res[1]))
        self.assertTrue(np.allclose(np.arange(2)*2, res[2]))

        res = v + 5.0
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) + 5.0, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) + 5.0, res[1]))
        self.assertTrue(np.allclose(np.arange(2) + 5.0, res[2]))

        res = 5.0 + v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) + 5.0, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) + 5.0, res[1]))
        self.assertTrue(np.allclose(np.arange(2) + 5.0, res[2]))

        with self.assertRaises(Exception) as context:
            res = v + bv.flatten()
        with self.assertRaises(Exception) as context:
            res = bv.flatten() + v

    def test_sub(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        res = v - v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4), res[1]))
        self.assertTrue(np.allclose(np.zeros(2), res[2]))

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])

        res = bv - v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4), res[1]))
        self.assertTrue(np.allclose(np.zeros(2), res[2]))

        res = v - bv
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4), res[1]))
        self.assertTrue(np.allclose(np.zeros(2), res[2]))

        res = 5.0 - v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(5.0 - np.arange(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(5.0 - np.arange(4), res[1]))
        self.assertTrue(np.allclose(5.0 - np.arange(2), res[2]))

        res = v - 5.0
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) - 5.0, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) - 5.0, res[1]))
        self.assertTrue(np.allclose(np.arange(2) - 5.0, res[2]))

        with self.assertRaises(Exception) as context:
            res = v - bv.flatten()
        with self.assertRaises(Exception) as context:
            res = bv.flatten() - v

    def test_mul(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        res = v * v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) * np.arange(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) * np.arange(4), res[1]))
        self.assertTrue(np.allclose(np.arange(2) * np.arange(2), res[2]))

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])

        res = v * bv
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) * np.arange(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) * np.arange(4), res[1]))
        self.assertTrue(np.allclose(np.arange(2) * np.arange(2), res[2]))

        res = bv * v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) * np.arange(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) * np.arange(4), res[1]))
        self.assertTrue(np.allclose(np.arange(2) * np.arange(2), res[2]))

        res = v * 2.0
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) * 2.0, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) * 2.0, res[1]))
        self.assertTrue(np.allclose(np.arange(2) * 2.0, res[2]))

        res = 2.0 * v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(3) * 2.0, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.arange(4) * 2.0, res[1]))
        self.assertTrue(np.allclose(np.arange(2) * 2.0, res[2]))

        with self.assertRaises(Exception) as context:
            res = v * bv.flatten()
        with self.assertRaises(Exception) as context:
            res = bv.flatten() * v

    def test_truediv(self):
        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 1.0
        if rank == 1:
            v[1] = np.arange(4) + 1.0
        v[2] = np.arange(2) + 1.0

        res = v / v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4), res[1]))
        self.assertTrue(np.allclose(np.ones(2), res[2]))

        bv = BlockVector([np.arange(3) + 1.0,
                          np.arange(4) + 1.0,
                          np.arange(2) + 1.0])

        res = v / bv
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4), res[1]))
        self.assertTrue(np.allclose(np.ones(2), res[2]))

        res = bv / v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4), res[1]))
        self.assertTrue(np.allclose(np.ones(2), res[2]))

        res = v / 2.0
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose((np.arange(3) + 1.0)/2.0, res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose((np.arange(4) + 1.0)/2.0, res[1]))
        self.assertTrue(np.allclose((np.arange(2) + 1.0)/2.0, res[2]))

        res = 2.0 / v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(2.0/(np.arange(3) + 1.0), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(2.0/(np.arange(4) + 1.0), res[1]))
        self.assertTrue(np.allclose(2.0/(np.arange(2) + 1.0), res[2]))

    def test_floordiv(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 1.0
        if rank == 1:
            v[1] = np.arange(4) + 1.0
        v[2] = np.arange(2) + 1.0

        res = v // v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4), res[1]))
        self.assertTrue(np.allclose(np.ones(2), res[2]))

        bv = BlockVector([np.arange(3) + 1.0,
                          np.arange(4) + 1.0,
                          np.arange(2) + 1.0])

        res = v // bv
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4), res[1]))
        self.assertTrue(np.allclose(np.ones(2), res[2]))

        res = bv // v
        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4), res[1]))
        self.assertTrue(np.allclose(np.ones(2), res[2]))

        res1 = v // 2.0
        res2 = bv // 2.0
        self.assertTrue(isinstance(res1, MPIBlockVector))
        self.assertEqual(3, res1.nblocks)
        self.assertTrue(np.allclose(res1.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res1.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(res1[0], res2[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res1.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(res1[1], res2[1]))
        self.assertTrue(np.allclose(res1[2], res2[2]))

        res1 = 2.0 // v
        res2 = 2.0 // bv
        self.assertTrue(isinstance(res1, MPIBlockVector))
        self.assertEqual(3, res1.nblocks)
        self.assertTrue(np.allclose(res1.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res1.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(res1[0], res2[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res1.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(res1[1], res2[1]))
        self.assertTrue(np.allclose(res1[2], res2[2]))

    def test_isum(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        v += v
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) * 2.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) * 2.0, v[1]))
        self.assertTrue(np.allclose(np.arange(2) * 2.0, v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])
        v += bv
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) * 2.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) * 2.0, v[1]))
        self.assertTrue(np.allclose(np.arange(2) * 2.0, v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3, dtype='d')
        if rank == 1:
            v[1] = np.arange(4, dtype='d')
        v[2] = np.arange(2, dtype='d')

        v += 7.0
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) + 7.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) + 7.0, v[1]))
        self.assertTrue(np.allclose(np.arange(2) + 7.0, v[2]))

    def test_isub(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        v -= v
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.zeros(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.zeros(4), v[1]))
        self.assertTrue(np.allclose(np.zeros(2), v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])
        v -= bv
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.zeros(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.zeros(4), v[1]))
        self.assertTrue(np.allclose(np.zeros(2), v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3, dtype='d')
        if rank == 1:
            v[1] = np.arange(4, dtype='d')
        v[2] = np.arange(2, dtype='d')

        v -= 7.0
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) - 7.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) - 7.0, v[1]))
        self.assertTrue(np.allclose(np.arange(2) - 7.0, v[2]))

    def test_imul(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        v *= v
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) * np.arange(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) * np.arange(4), v[1]))
        self.assertTrue(np.allclose(np.arange(2) * np.arange(2), v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])
        v *= bv
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) * np.arange(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) * np.arange(4), v[1]))
        self.assertTrue(np.allclose(np.arange(2) * np.arange(2), v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3, dtype='d')
        if rank == 1:
            v[1] = np.arange(4, dtype='d')
        v[2] = np.arange(2, dtype='d')

        v *= 7.0
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) * 7.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) * 7.0, v[1]))
        self.assertTrue(np.allclose(np.arange(2) * 7.0, v[2]))

    def test_itruediv(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 1.0
        if rank == 1:
            v[1] = np.arange(4) + 1.0
        v[2] = np.arange(2) + 1.0

        v /= v
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.ones(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.ones(4), v[1]))
        self.assertTrue(np.allclose(np.ones(2), v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3) + 1.0
        if rank == 1:
            v[1] = np.arange(4) + 1.0
        v[2] = np.arange(2) + 1.0

        bv = BlockVector([np.arange(3) + 1.0,
                          np.arange(4) + 1.0,
                          np.arange(2) + 1.0])
        v /= bv
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.ones(3), v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.ones(4), v[1]))
        self.assertTrue(np.allclose(np.ones(2), v[2]))

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3, dtype='d')
        if rank == 1:
            v[1] = np.arange(4, dtype='d')
        v[2] = np.arange(2, dtype='d')

        v /= 2.0
        self.assertTrue(isinstance(v, MPIBlockVector))
        self.assertEqual(3, v.nblocks)
        if rank == 0:
            self.assertTrue(np.allclose(np.arange(3) / 2.0, v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(np.arange(4) / 2.0, v[1]))
        self.assertTrue(np.allclose(np.arange(2) / 2.0, v[2]))

    def test_le(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 8
        if rank == 1:
            v[1] = np.ones(4) * 2
        v[2] = np.ones(2) * 4

        v1 = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3) * 2
        if rank == 1:
            v1[1] = np.ones(4) * 8
        v1[2] = np.ones(2) * 4

        res = v <= v1

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

        bv = BlockVector([np.ones(3) * 2,
                          np.ones(4) * 8,
                          np.ones(2) * 4])

        res = v <= bv

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

        res = v <= 3.0

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

    def test_lt(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 8
        if rank == 1:
            v[1] = np.ones(4) * 2
        v[2] = np.ones(2) * 4

        v1 = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3) * 2
        if rank == 1:
            v1[1] = np.ones(4) * 8
        v1[2] = np.ones(2) * 4

        res = v < v1

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

        bv = BlockVector([np.ones(3) * 2,
                          np.ones(4) * 8,
                          np.ones(2) * 4])

        res = v < bv

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

        res = v <= 3.0

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

    def test_ge(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 8
        if rank == 1:
            v[1] = np.ones(4) * 2
        v[2] = np.ones(2) * 4

        v1 = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3) * 2
        if rank == 1:
            v1[1] = np.ones(4) * 8
        v1[2] = np.ones(2) * 4

        res = v >= v1

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

        bv = BlockVector([np.ones(3) * 2,
                          np.ones(4) * 8,
                          np.ones(2) * 4])

        res = v >= bv

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

        res = v >= 3.0

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

    def test_gt(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 8
        if rank == 1:
            v[1] = np.ones(4) * 2
        v[2] = np.ones(2) * 4

        v1 = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3) * 2
        if rank == 1:
            v1[1] = np.ones(4) * 8
        v1[2] = np.ones(2) * 4

        res = v > v1

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

        bv = BlockVector([np.ones(3) * 2,
                          np.ones(4) * 8,
                          np.ones(2) * 4])

        res = v > bv

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

        res = v > 3.0

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

    def test_eq(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 8
        if rank == 1:
            v[1] = np.ones(4) * 2
        v[2] = np.ones(2) * 4

        v1 = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3) * 2
        if rank == 1:
            v1[1] = np.ones(4) * 8
        v1[2] = np.ones(2) * 4

        res = v == v1

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

        bv = BlockVector([np.ones(3) * 2,
                          np.ones(4) * 8,
                          np.ones(2) * 4])

        res = v == bv

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

        res = v == 8.0

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

    def test_ne(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 8
        if rank == 1:
            v[1] = np.ones(4) * 2
        v[2] = np.ones(2) * 4

        v1 = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v1[0] = np.ones(3) * 2
        if rank == 1:
            v1[1] = np.ones(4) * 8
        v1[2] = np.ones(2) * 4

        res = v != v1

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

        bv = BlockVector([np.ones(3) * 2,
                          np.ones(4) * 8,
                          np.ones(2) * 4])

        res = v != bv

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.zeros(2, dtype=bool), res[2]))

        res = v != 8.0

        self.assertTrue(isinstance(res, MPIBlockVector))
        self.assertEqual(3, res.nblocks)
        self.assertTrue(np.allclose(res.shared_blocks, v.shared_blocks))

        if rank == 0:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.zeros(3, dtype=bool), res[0]))
        if rank == 1:
            self.assertTrue(np.allclose(res.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(np.ones(4, dtype=bool), res[1]))
        self.assertTrue(np.allclose(np.ones(2, dtype=bool), res[2]))

    def test_unary_ufuncs(self):

        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 0.5
        if rank == 1:
            v[1] = np.ones(2) * 0.8

        bv = BlockVector(2)
        a = np.ones(3) * 0.5
        b = np.ones(2) * 0.8
        bv[0] = a
        bv[1] = b

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

        bv2 = BlockVector(2)
        for fun in unary_funcs:
            bv2[0] = fun(bv[0])
            bv2[1] = fun(bv[1])
            res = fun(v)
            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], bv2[i]))

        with self.assertRaises(Exception) as context:
            np.cbrt(v)

        with self.assertRaises(Exception) as context:
            np.cumsum(v)

        with self.assertRaises(Exception) as context:
            np.cumprod(v)

        with self.assertRaises(Exception) as context:
            np.cumproduct(v)

    def test_reduce_ufuncs(self):

        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 0.5
        if rank == 1:
            v[1] = np.ones(2) * 0.8

        bv = BlockVector(2)
        bv[0] = np.ones(3) * 0.5
        bv[1] = np.ones(2) * 0.8

        reduce_funcs = [np.sum, np.max, np.min, np.prod]
        for fun in reduce_funcs:
            self.assertAlmostEqual(fun(v), fun(bv.flatten()))

        with self.assertRaises(Exception) as context:
            np.mean(v)

        other_funcs = [np.all, np.any]
        for fun in other_funcs:
            self.assertAlmostEqual(fun(v), fun(bv.flatten()))

    def test_binary_ufuncs(self):

        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3) * 0.5
        if rank == 1:
            v[1] = np.ones(2) * 0.8

        v2 = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v2[0] = np.ones(3) * 3.0
        if rank == 1:
            v2[1] = np.ones(2) * 2.8

        bv = BlockVector(2)
        bv[0] = np.ones(3) * 0.5
        bv[1] = np.ones(2) * 0.8

        bv2 = BlockVector(2)
        bv2[0] = np.ones(3) * 3.0
        bv2[1] = np.ones(2) * 2.8

        binary_ufuncs = [np.add, np.multiply, np.divide, np.subtract,
                         np.greater, np.greater_equal, np.less,
                         np.less_equal, np.not_equal,
                         np.maximum, np.minimum,
                         np.fmax, np.fmin, np.equal,
                         np.logaddexp, np.logaddexp2, np.remainder,
                         np.heaviside, np.hypot]

        for fun in binary_ufuncs:
            serial_res = fun(bv, bv2)
            res = fun(v, v2)

            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))

            serial_res = fun(bv, bv2)
            res = fun(v, bv2)

            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))

            serial_res = fun(bv, bv2)
            res = fun(bv, v2)

            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))

            serial_res = fun(bv, 2.0)
            res = fun(v, 2.0)

            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))

            serial_res = fun(2.0, bv)
            res = fun(2.0, v)

            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))


        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3, dtype=bool)
        if rank == 1:
            v[1] = np.ones(2, dtype=bool)

        v2 = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v2[0] = np.zeros(3, dtype=bool)
        if rank == 1:
            v2[1] = np.zeros(2, dtype=bool)

        bv = BlockVector(2)
        bv[0] = np.ones(3, dtype=bool)
        bv[1] = np.ones(2, dtype=bool)

        bv2 = BlockVector(2)
        bv2[0] = np.zeros(3, dtype=bool)
        bv2[1] = np.zeros(2, dtype=bool)

        binary_ufuncs = [np.logical_and, np.logical_or, np.logical_xor]
        for fun in binary_ufuncs:
            serial_res = fun(bv, bv2)
            res = fun(v, v2)
            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))

            serial_res = fun(bv, bv2)
            res = fun(v, bv2)
            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))

            serial_res = fun(bv, bv2)
            res = fun(bv, v2)
            self.assertIsInstance(res, MPIBlockVector)
            self.assertEqual(res.nblocks, 2)
            for i in res.owned_blocks:
                self.assertTrue(np.allclose(res[i], serial_res[i]))


    def test_contains(self):

        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3)
        if rank == 1:
            v[1] = np.zeros(2)

        self.assertTrue(0 in v)
        self.assertFalse(3 in v)

    def test_len(self):

        v = MPIBlockVector(2, [0,1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.ones(3)
        if rank == 1:
            v[1] = np.zeros(2)

        self.assertEqual(len(v), 5)

    def test_copyfrom(self):

        v = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            v[0] = np.arange(3)
        if rank == 1:
            v[1] = np.arange(4)
        v[2] = np.arange(2)

        bv = BlockVector([np.arange(3), np.arange(4), np.arange(2)])
        vv = MPIBlockVector(3, [0,1,-1], comm)
        vv.copyfrom(v)

        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))

        vv = MPIBlockVector(3, [0,1,-1], comm)
        vv.copyfrom(bv)

        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))

        vv = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            vv[0] = np.arange(3) + 1
        if rank == 1:
            vv[1] = np.arange(4) + 1
        vv[2] = np.arange(2) + 1

        vv.copyfrom(bv)

        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))

        vv = MPIBlockVector(3, [0,1,-1], comm)
        rank = comm.Get_rank()
        if rank == 0:
            vv[0] = np.arange(3) + 1
        if rank == 1:
            vv[1] = np.arange(4) + 1
        vv[2] = np.arange(2) + 1

        vv.copyfrom(v)

        self.assertTrue(isinstance(vv, MPIBlockVector))
        self.assertEqual(vv.nblocks, v.nblocks)
        self.assertTrue(np.allclose(vv.shared_blocks, v.shared_blocks))
        if rank == 0:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[0], v[0]))
        if rank == 1:
            self.assertTrue(np.allclose(vv.owned_blocks, v.owned_blocks))
            self.assertTrue(np.allclose(vv[1], v[1]))
        self.assertTrue(np.allclose(vv[2], v[2]))
