#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2025
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available

if not numpy_available:
    raise unittest.SkipTest('pynumero MA27 tests require numpy')
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface


@unittest.skipIf(not MA57Interface.available(), reason='MA57 not available')
class TestMA57Interface(unittest.TestCase):
    def test_get_cntl(self):
        ma57 = MA57Interface()
        self.assertEqual(ma57.get_icntl(1), 6)
        self.assertEqual(ma57.get_icntl(7), 1)

        self.assertAlmostEqual(ma57.get_cntl(1), 1e-2)  # Numerical pivot threshold
        self.assertAlmostEqual(ma57.get_cntl(2), 1e-20)  # Null pivot threshold

    def test_set_icntl(self):
        ma57 = MA57Interface()
        ma57.set_icntl(5, 4)  # Set output printing to max verbosity
        ma57.set_icntl(8, 1)  # Keep factors when we run out of space
        # (so MA57ED can be used)
        icntl5 = ma57.get_icntl(5)
        icntl8 = ma57.get_icntl(8)
        self.assertEqual(icntl5, 4)
        self.assertEqual(icntl8, 1)

        with self.assertRaisesRegex(TypeError, 'must be an integer'):
            ma57.set_icntl(1.0, 0)
        with self.assertRaisesRegex(IndexError, 'is out of range'):
            ma57.set_icntl(100, 0)
        with self.assertRaises(ctypes.ArgumentError):
            ma57.set_icntl(1, 0.0)

    def test_set_cntl(self):
        ma57 = MA57Interface()
        ma57.set_cntl(1, 1e-8)
        ma57.set_cntl(2, 1e-12)
        self.assertAlmostEqual(ma57.get_cntl(1), 1e-8)
        self.assertAlmostEqual(ma57.get_cntl(2), 1e-12)

    def test_do_symbolic_factorization(self):
        ma57 = MA57Interface()

        n = 5
        ne = 7
        irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
        jcn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
        # Copied these Fortran-style indices from HSL docs.
        # Interface expects C-style indices, as is typical in Python.
        irn = irn - 1
        jcn = jcn - 1

        bad_jcn = np.array([1, 2, 3, 5, 3, 4], dtype=np.intc)

        ma57.do_symbolic_factorization(n, irn, jcn)

        self.assertEqual(ma57.get_info(1), 0)
        self.assertEqual(ma57.get_info(4), 0)
        self.assertEqual(ma57.get_info(9), 48)  # Min required length of FACT
        self.assertEqual(ma57.get_info(10), 53)  # Min required length of IFACT
        self.assertEqual(ma57.get_info(14), 0)  # Should not yet be set

        with self.assertRaisesRegex(AssertionError, 'Dimension mismatch'):
            ma57.do_symbolic_factorization(n, irn, bad_jcn)

    def test_do_numeric_factorization(self):
        ma57 = MA57Interface()

        n = 5
        ne = 7
        irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
        jcn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
        irn = irn - 1
        jcn = jcn - 1
        ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0], dtype=np.double)
        ma57.do_symbolic_factorization(n, irn, jcn)
        ma57.fact_factor = 1.5
        ma57.ifact_factor = 1.5
        # ^ No way to check whether these are handled properly... Would have to
        # access the struct to get LFACT, LIFACT

        status = ma57.do_numeric_factorization(n, ent)
        self.assertEqual(status, 0)

        self.assertEqual(ma57.get_info(14), 12)  # 12 entries in factors
        self.assertEqual(ma57.get_info(24), 2)  # 2 negative eigenvalues
        self.assertEqual(ma57.get_info(22), 1)  # 1 2x2 pivot
        self.assertEqual(ma57.get_info(23), 0)  # 0 delayed pivots

        ent2 = np.array([1.0, 5.0, 1.0, 6.0, 4.0, 3.0, 2.0], dtype=np.double)
        ma57.do_numeric_factorization(n, ent2)
        self.assertEqual(status, 0)

        bad_ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0], dtype=np.double)
        with self.assertRaisesRegex(AssertionError, 'Wrong number of entries'):
            ma57.do_numeric_factorization(n, bad_ent)
        with self.assertRaisesRegex(AssertionError, 'Dimension mismatch'):
            ma57.do_numeric_factorization(n + 1, ent)

        n = 5
        ne = 8
        irn = np.array([1, 1, 2, 2, 3, 3, 5, 5], dtype=np.intc)
        jcn = np.array([1, 2, 3, 5, 3, 4, 5, 1], dtype=np.intc)
        irn = irn - 1
        jcn = jcn - 1
        ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0, -1.3], dtype=np.double)
        status = ma57.do_symbolic_factorization(n, irn, jcn)
        self.assertEqual(status, 0)
        status = ma57.do_numeric_factorization(n, ent)
        self.assertEqual(status, 0)
        self.assertEqual(ma57.get_info(24), 2)
        self.assertEqual(ma57.get_info(23), 0)

    def test_do_backsolve(self):
        ma57 = MA57Interface()

        n = 5
        ne = 7
        irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
        jcn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
        irn = irn - 1
        jcn = jcn - 1
        ent = np.array([2.0, 3.0, 4.0, 6.0, 1.0, 5.0, 1.0], dtype=np.double)
        rhs = np.array([8.0, 45.0, 31.0, 15.0, 17.0], dtype=np.double)
        status = ma57.do_symbolic_factorization(n, irn, jcn)
        status = ma57.do_numeric_factorization(n, ent)
        sol = ma57.do_backsolve(rhs)

        expected_sol = [1, 2, 3, 4, 5]
        old_rhs = np.array([8.0, 45.0, 31.0, 15.0, 17.0])
        for i in range(n):
            self.assertAlmostEqual(sol[i], expected_sol[i])
            self.assertEqual(old_rhs[i], rhs[i])

        # rhs2 = np.array([[8., 17.],
        #                 [45., 15.],
        #                 [31., 31.],
        #                 [15., 45.],
        #                 [17., 8.]], dtype=np.double)
        # sol = ma57.do_backsolve(rhs2)
        # FIXME
        # This gives unexpected (incorrect) results.
        # Need to investigate further.


if __name__ == '__main__':
    unittest.main()
