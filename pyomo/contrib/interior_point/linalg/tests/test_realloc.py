import pyutilib.th as unittest
from pyomo.common.dependencies import attempt_import

np, numpy_available = attempt_import('numpy', 'Interior point requires numpy',
        minimum_version='1.13.0')
scipy, scipy_available = attempt_import('scipy', 'Interior point requires scipy')
mumps_interface, mumps_available = attempt_import(
        'pyomo.contrib.interior_point.linalg.mumps_interface',
        'Interior point requires mumps')
if not (numpy_available and scipy_available):
    raise unittest.SkipTest('Interior point tests require numpy and scipy')

from pyomo.contrib.pynumero.extensions.asl import AmplInterface
asl_available = AmplInterface.available()

from pyomo.contrib.interior_point.interior_point import InteriorPointSolver
from pyomo.contrib.interior_point.interface import InteriorPointInterface

from scipy.sparse import coo_matrix


class TestReallocation(unittest.TestCase):

    @unittest.skipIf(not mumps_available, 'mumps is not available')
    def test_reallocate_memory_mumps(self):

        # Create a tri-diagonal matrix with small entries on the diagonal
        n = 10000
        small_val = 1e-7
        big_val = 1e2
        irn = []
        jcn = []
        ent = []
        for i in range(n-1):
            irn.extend([i+1, i, i])
            jcn.extend([i, i, i+1])
            ent.extend([big_val,small_val,big_val])
        irn.append(n-1)
        jcn.append(n-1)
        ent.append(small_val)
        irn = np.array(irn)
        jcn = np.array(jcn)
        ent = np.array(ent)

        matrix = coo_matrix((ent, (irn, jcn)), shape=(n,n))

        linear_solver = mumps_interface.MumpsInterface()
        linear_solver.do_symbolic_factorization(matrix)

        predicted = linear_solver.get_infog(16)

        with self.assertRaisesRegex(RuntimeError, 'MUMPS error: -9'):
            linear_solver.do_numeric_factorization(matrix)

        linear_solver.do_symbolic_factorization(matrix)

        factor = 2
        linear_solver.increase_memory_allocation(factor)

        linear_solver.do_numeric_factorization(matrix)

        # Expected memory allocation (MB)
        self.assertEqual(linear_solver._prev_allocation, 6)

        actual = linear_solver.get_infog(18)

        # Sanity checks:
        # Make sure actual memory usage is greater than initial guess
        self.assertTrue(predicted < actual)
        # Make sure memory allocation is at least as much as was used
        self.assertTrue(actual <= linear_solver._prev_allocation)


if __name__ == '__main__':
    test_realloc = TestReallocation()
    test_realloc.test_reallocate_memory_mumps()
